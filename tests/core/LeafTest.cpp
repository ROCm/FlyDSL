//===- LeafTest.cpp - Leaf reference results -------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Scalar algebra of Leaf<Runtime>, through the single-leaf IntTuple operators and
// leaf_* functions.  A result is static exactly when both operands are or when
// a fold applies (`_0 * x`, `_0 / x`, `x % _1`, `_0 % x`, `_0 & x`).  Divisibility:
// `*` multiplies, `// k` keeps div/k when k divides div, and `% k` is a static 0
// when k divides div and otherwise keeps gcd(div, k); `+` and `-` keep the gcd.
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

using namespace core_test;

TEST(LeafParity, AddSub) {
  EXPECT_EQ(toString(3_ct + 4_ct), "7");
  EXPECT_EQ(toString(3_dt + 4_ct), "?7");
  EXPECT_EQ(toString(3_dt + 4_dt), "?7");
  // Adding a static zero still yields a runtime value.
  EXPECT_EQ(toString(3_dt + 0_ct), "?3");
  EXPECT_EQ(toString(dyn(8, 4) + 6_ct), "?14{div=2}");
  EXPECT_EQ(toString(dyn(8, 8) + dyn(12, 12)), "?20{div=4}");
  EXPECT_EQ(toString(3_dt - 4_ct), "?-1");
  EXPECT_EQ(toString(dyn(24, 8) - dyn(12, 4)), "?12{div=4}");
  EXPECT_EQ(toString(-3_ct), "-3");
  EXPECT_EQ(toString(-3_dt), "?-3");
  EXPECT_EQ(toString(dyn(8, 8) + dyn(16, 8)), "?24{div=8}");
  // Subtracting equal runtime values still yields a runtime value.
  EXPECT_EQ(toString(dyn(16, 8) - dyn(16, 8)), "?0{div=8}");
}

TEST(LeafParity, Mul) {
  EXPECT_EQ(toString(3_ct * 4_ct), "12");
  EXPECT_EQ(toString(3_dt * 4_ct), "?12{div=4}");
  EXPECT_EQ(toString(3_dt * 0_ct), "0");
  EXPECT_EQ(toString(0_ct * 3_dt), "0");
  EXPECT_EQ(toString(3_dt * 1_ct), "?3");
  EXPECT_EQ(toString(dyn(8, 4) * dyn(6, 2)), "?48{div=8}");
  EXPECT_EQ(toString(dyn(8, 4) * 3_ct), "?24{div=12}");
}

TEST(LeafParity, Div) {
  EXPECT_EQ(toString(7_ct / 2_ct), "3");
  EXPECT_EQ(toString(7_dt / 2_ct), "?3");
  EXPECT_EQ(toString(0_ct / 7_dt), "0");
  EXPECT_EQ(toString(7_dt / 1_ct), "?7");
  EXPECT_EQ(toString(dyn(16, 8) / 4_ct), "?4{div=2}");
  EXPECT_EQ(toString(dyn(16, 8) / 3_ct), "?5");
  EXPECT_EQ(toString(dyn(16, 8) / dyn(4, 4)), "?4");
  // C++ division truncates toward zero.
  EXPECT_EQ(toString(-7_ct / 2_ct), "-3");
  EXPECT_EQ(toString(7_ct / -2_ct), "-3");
  EXPECT_EQ(toString(-7_ct / -2_ct), "3");
  EXPECT_EQ(toString(-7_dt / 2_ct), "?-3");
  EXPECT_EQ(toString(7_dt / -2_ct), "?-3");
}

TEST(LeafParity, Mod) {
  EXPECT_EQ(toString(7_ct % 3_ct), "1");
  EXPECT_EQ(toString(7_dt % 3_ct), "?1");
  EXPECT_EQ(toString(7_dt % 1_ct), "0");
  EXPECT_EQ(toString(0_ct % 7_dt), "0");
  // The divisibility makes the remainder static.
  EXPECT_EQ(toString(dyn(16, 8) % 4_ct), "0");
  EXPECT_EQ(toString(dyn(18, 6) % 4_ct), "?2{div=2}");
  // The remainder takes the sign of the dividend.
  EXPECT_EQ(toString(-7_ct % 2_ct), "-1");
  EXPECT_EQ(toString(7_ct % -2_ct), "1");
  EXPECT_EQ(toString(-7_ct % -2_ct), "-1");
  EXPECT_EQ(toString(-7_dt % 2_ct), "?-1");
  EXPECT_EQ(toString(dyn(24, 8) % 12_ct), "?0{div=4}");
  EXPECT_EQ(toString(dyn(16, 8) % dyn(12, 4)), "?4{div=4}");
}

TEST(LeafParity, MinMaxGcd) {
  EXPECT_EQ(toString(leaf_min(3_ct, 4_ct)), "3");
  EXPECT_EQ(toString(leaf_min(3_dt, 4_ct)), "?3");
  EXPECT_EQ(toString(leaf_max(3_dt, 4_ct)), "?4");
  EXPECT_EQ(toString(leaf_gcd(6_ct, 4_ct)), "2");
  EXPECT_EQ(toString(leaf_gcd(6_dt, 4_ct)), "?2");
  EXPECT_EQ(toString(leaf_min(-7_dt, 2_ct)), "?-7");
  EXPECT_EQ(toString(leaf_gcd(12_ct, 0_ct)), "12");
  EXPECT_EQ(toString(leaf_min(dyn(8, 4), dyn(12, 6))), "?8{div=2}");
  EXPECT_EQ(toString(leaf_max(dyn(8, 4), 8_ct)), "?8{div=4}");
  EXPECT_EQ(toString(leaf_min(dyn(8, 4), 6_ct)), "?6{div=2}");
  EXPECT_EQ(toString(leaf_min(dyn(8, 8), dyn(16, 16))), "?8{div=8}");
  EXPECT_EQ(toString(leaf_gcd(dyn(8, 4), dyn(12, 6))), "?4{div=2}");
  EXPECT_EQ(toString(leaf_gcd(dyn(16, 8), 12_ct)), "?4{div=4}");
  EXPECT_EQ(toString(leaf_gcd(dyn(8, 4), 0_ct)), "?8{div=4}");
  // A multiple of 16 and 8 always have gcd 8.
  EXPECT_EQ(toString(leaf_gcd(dyn(16, 16), 8_ct)), "8");
}

TEST(LeafParity, ShapeArithmetic) {
  EXPECT_EQ(toString(leaf_ceil_div(7_ct, 2_ct)), "4");
  EXPECT_EQ(toString(leaf_ceil_div(7_dt, 2_ct)), "?4");
  EXPECT_EQ(toString(leaf_shape_div(12_ct, 4_ct)), "3");
  EXPECT_EQ(toString(leaf_shape_div(12_dt, 4_ct)), "?3");
  EXPECT_EQ(toString(leaf_shape_div(4_ct, 12_dt)), "?1");
  EXPECT_EQ(toString(leaf_safe_div(12_ct, 4_ct)), "3");
  EXPECT_EQ(toString(leaf_safe_div(3_ct, 4_ct)), "!inexact-division");
  // With a dynamic operand safe_div is the plain quotient, exact or not.
  EXPECT_EQ(toString(leaf_safe_div(12_dt, 4_ct)), "?3");
  EXPECT_EQ(toString(leaf_safe_div(13_dt, 4_ct)), "?3");
  EXPECT_EQ(toString(leaf_safe_div(12_ct, 5_dt)), "?2");
  EXPECT_EQ(toString(leaf_safe_div(dyn(12, 4), 4_ct)), "?3");
  EXPECT_EQ(toString(leaf_abs(-3_dt)), "?3");
  EXPECT_EQ(toString(leaf_signum(-3_dt)), "?-1");
  EXPECT_EQ(toString(leaf_abs(-7_ct)), "7");
  EXPECT_EQ(toString(leaf_signum(0_ct)), "0");
  EXPECT_EQ(toString(leaf_shape_div(0_ct, 3_ct)), "0");
  EXPECT_EQ(toString(leaf_ceil_div(dyn(18, 6), 4_ct)), "?5");
  EXPECT_EQ(toString(leaf_shape_div(dyn(16, 8), dyn(4, 4))), "?4");
  EXPECT_EQ(toString(leaf_safe_div(dyn(16, 8), 4_ct)), "?4{div=2}");
}

// A cast keeps the value. The Runtime category checks that the width holds it.
TEST(LeafParity, Cast) {
  EXPECT_EQ(toString(leaf_cast(3_dt, LogWidth::I64)), "?3i64");
  EXPECT_EQ(toString(leaf_cast(-3_dt, LogWidth::I64)), "?-3i64");
  EXPECT_EQ(toString(leaf_cast(dyn(24, 8), LogWidth::I64)), "?24i64{div=8}");
  auto wide = [](int64_t value) { return Tup::fromLeaf(RT::getDyn(value, LogWidth::I64)); };
  EXPECT_EQ(toString(leaf_cast(wide(-7), LogWidth::I32)), "?-7");
  EXPECT_EQ(toString(leaf_cast(wide(INT32_MAX), LogWidth::I32)), "?2147483647");
  EXPECT_EQ(toString(leaf_cast(wide(int64_t{1} << 31), LogWidth::I32)), "!integer-overflow");
  EXPECT_EQ(toString(leaf_cast(300_dt, LogWidth::I8)), "!integer-overflow");
  // A static integer stays static.
  EXPECT_EQ(toString(leaf_cast(7_ct, LogWidth::I64)), "7");
  EXPECT_EQ(toString(leaf_cast(8589934592_ct, LogWidth::I32)), "!integer-overflow");
  // The sum of two i32 values that passes 2^31 is exact at i64.
  auto big = dyn(INT32_MAX);
  EXPECT_EQ(toString(leaf_add(big, big)), "!integer-overflow");
  EXPECT_EQ(toString(leaf_add(leaf_cast(big, LogWidth::I64), leaf_cast(big, LogWidth::I64))),
            "?4294967294i64");
  EXPECT_EQ(toString(leaf_cast(make_tuple(1_c, 2_c), LogWidth::I64)), "!expected-leaf-operand");
}

TEST(LeafParity, Comparison) {
  EXPECT_EQ(toString(leaf_lt(3_ct, 4_ct)), "1");
  EXPECT_EQ(toString(leaf_lt(3_dt, 4_ct)), "?1");
  EXPECT_EQ(toString(leaf_eq(3_dt, 3_ct)), "?1");
  EXPECT_EQ(toString(leaf_lt(dyn(8, 4), 6_ct)), "?0");
  // The divisibility does not decide a comparison.
  EXPECT_EQ(toString(leaf_eq(dyn(8, 4), 6_ct)), "?0");
}

TEST(LeafParity, Bitwise) {
  EXPECT_EQ(toString(leaf_bit_and(6_dt, 3_ct)), "?2");
  EXPECT_EQ(toString(leaf_bit_and(6_dt, 0_ct)), "0");
  EXPECT_EQ(toString(leaf_bit_or(6_dt, 3_ct)), "?7");
  EXPECT_EQ(toString(leaf_bit_or(6_dt, 0_ct)), "?6");
  EXPECT_EQ(toString(leaf_bit_xor(6_dt, 3_ct)), "?5");
  EXPECT_EQ(toString(leaf_shl(3_dt, 2_ct)), "?12{div=4}");
  EXPECT_EQ(toString(leaf_shr(12_dt, 2_ct)), "?3");
  // >> of a negative value shifts in its sign.
  EXPECT_EQ(toString(leaf_shr(-7_dt, 2_ct)), "?-2");
  // A negative count shifts the other way.
  EXPECT_EQ(toString(leaf_shl(8_ct, -3_ct)), "1");
  EXPECT_EQ(toString(leaf_shr(8_ct, -3_ct)), "64");
  EXPECT_EQ(toString(leaf_shl(-7_dt, -2_ct)), "?-2");
  EXPECT_EQ(toString(leaf_shr(-7_dt, -2_ct)), "?-28{div=4}");
  EXPECT_EQ(toString(leaf_shl(dyn(8, 4), 2_ct)), "?32{div=16}");
  EXPECT_EQ(toString(leaf_shr(dyn(32, 16), 2_ct)), "?8{div=4}");
  EXPECT_EQ(toString(leaf_shr(dyn(24, 8), 4_ct)), "?1");
  // The divisibility clears every bit of the mask, which makes the result static.
  EXPECT_EQ(toString(leaf_bit_and(dyn(16, 8), 7_ct)), "0");
}

// Cases where Fly does not reproduce the reference; the expected results are the reference's.  Run
// with --gtest_also_run_disabled_tests to see them.
TEST(LeafParity, DISABLED_KnownDivergences) {
  // CuTe C++ returns a runtime 0; Fly folds shape_div(0, b) to a static 0.
  EXPECT_EQ(toString(leaf_shape_div(0_ct, 12_dt)), "?0");
  // CuTe C++ returns a runtime 1 for gcd with a dynamic operand; Fly folds it to a static 1.
  EXPECT_EQ(toString(leaf_gcd(1_ct, 6_dt)), "?1");
  // CuTe C++ returns a runtime 1; Fly folds it to 1, the value for every b > 0.
  EXPECT_EQ(toString(leaf_ceil_div(1_ct, 7_dt)), "?1");
  // CuTe C++ returns a runtime 1; Fly folds it to 1, the value for every b > 0.
  EXPECT_EQ(toString(leaf_shape_div(1_ct, 12_dt)), "?1");
  // CuTe C++ gcd keeps the sign the Euclidean remainders leave; Fly returns the non-negative gcd.
  EXPECT_EQ(toString(leaf_gcd(-12_ct, 8_ct)), "-4");
  EXPECT_EQ(toString(leaf_gcd(0_ct, -12_ct)), "-12");
  EXPECT_EQ(toString(leaf_gcd(-12_dt, 8_ct)), "?-4");
}
