//===- SwizzleTest.cpp - Swizzle reference results -------------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Swizzle<B,M,S> of Swizzle.hpp: the swizzled offset, its static-ness, and the swizzle
// algebra.
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

using namespace core_test;

// The swizzle XORs bits [M+S, M+S+B) into bits [M, M+B); a dynamic offset gives a dynamic result.
TEST(SwizzleParity, Apply) {
  EXPECT_EQ(toString(Swz(3, 4, 3)(1000_ct)), "920");
  EXPECT_EQ(toString(Swz(3, 4, 3)(0_ct)), "0");
  EXPECT_EQ(toString(Swz(3, 4, 3)(1000_dt)), "?920");
  EXPECT_EQ(toString(Swz(3, 4, 3)(dyn(1024, 16))), "?1024{div=16}");
  EXPECT_EQ(toString(Swz(2, 0, -3)(37_ct)), "45");
  EXPECT_EQ(toString(Swz(0, 4, 3)(1000_dt)), "?1000");
  EXPECT_EQ(toString(Swz(3, 3, 4)(305419896_ct)), "305419864");
  EXPECT_EQ(toString(Swz(2, 0, -3)(610800463_ct)), "610800471");
  EXPECT_EQ(toString(Swz(4, 4, -11)(610800471_ct)), "610964311");
  EXPECT_EQ(toString(Swz(2, 0, -3)(37_dt)), "?45");
  // A dynamic offset keeps the divisibility of the low bits the XOR cannot change.
  EXPECT_EQ(toString(Swz(3, 4, 3)(dyn(1000, 8))), "?920{div=8}");
  EXPECT_EQ(toString(Swz(3, 4, 3)(dyn(1024, 64))), "?1024{div=16}");
  EXPECT_EQ(toString(Swz(3, 3, 4)(dyn(1024, 16))), "?1024{div=8}");
  EXPECT_EQ(toString(Swz(2, 0, -3)(dyn(36, 2))), "?36{div=2}");
  EXPECT_EQ(toString(Swz(2, 0, -3)(dyn(40, 8))), "?40{div=8}");
  EXPECT_EQ(toString(Swz(2, 3, -4)(dyn(40, 8))), "?168{div=8}");
  EXPECT_EQ(toString(Swz(0, 4, 3)(dyn(1024, 16))), "?1024{div=16}");
  // An offset with no bit under the mask is left unchanged.
  EXPECT_EQ(toString(Swz(3, 4, 3)(dyn(1024, 1024))), "?1024{div=1024}");
}

TEST(SwizzleParity, Algebra) {
  EXPECT_EQ(toString(*composition(Swz(2, 4, 3), Swz(1, 6, 3))), "S<3,4,3>");
  EXPECT_EQ(toString(right_inverse(Swz(3, 4, 3))), "S<3,4,3>");
  EXPECT_EQ(toString(left_inverse(Swz(3, 4, 3))), "S<3,4,3>");
  EXPECT_EQ(toString(max_alignment(Swz(3, 4, 3))), "16");
  EXPECT_EQ(toString(upcast(4, Swz(3, 4, 3))), "S<3,2,3>");
  EXPECT_EQ(toString(downcast(2, Swz(3, 4, 3))), "S<3,5,3>");
  EXPECT_EQ(toString(recast_layout(16, 32, Swz(3, 4, 3))), "S<3,5,3>");
  // When N exceeds 2^M, the mask bits below the new unit drop out: S<max(B+M-log2 N, 0),0,S>.
  EXPECT_EQ(toString(upcast(4, Swz(2, 1, 3))), "S<1,0,3>");
  EXPECT_EQ(toString(upcast(2, Swz(1, 0, 2))), "S<0,0,2>");
  // make_swizzle<YMask, ZMask>
  EXPECT_EQ(toString(*make_swizzle<Runtime>(896, 56)), "S<3,3,4>");
  EXPECT_FALSE(composition(Swz(1, 0, 4), Swz(1, 0, 3)).has_value());
  EXPECT_FALSE(composition(Swz(1, 0, 4), Swz(1, 2, 4)).has_value());
}
