//===- ComposedLayoutTest.cpp - ComposedLayout reference results -----*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// ComposedLayout (inner o offset o outer) of ComposedLayout.hpp.  Inputs are
// make_composed_layout(inner, offset, outer).
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

using namespace core_test;

// The inner function, the offset and the outer layout print in that order.
TEST(ComposedLayoutParity, Construction) {
  EXPECT_EQ(toString(make_composed_layout(
                Swz(3, 4, 3), 0_ct, make_layout(make_tuple(8_c, 64_c), make_tuple(64_c, 1_c)))),
            "S<3,4,3> o 0 o (8,64):(64,1)");
  EXPECT_EQ(toString(make_composed_layout(
                Swz(3, 4, 3), 32_dt, make_layout(make_tuple(8_c, 64_d), make_tuple(64_d, 1_c)))),
            "S<3,4,3> o ?32 o (8,?64):(?64,1)");
  EXPECT_EQ(toString(make_composed_layout(make_layout(make_tuple(4_c, 8_c), make_tuple(8_c, 1_c)),
                                          2_ct, make_layout(32_ct, 1_ct))),
            "(4,8):(8,1) o 2 o 32:1");
  EXPECT_EQ(toString(make_composed_layout(Swz(3, 4, 3), 0_ct,
                                          make_layout(make_tuple(8_c, make_tuple(4_c, 16_c)),
                                                      make_tuple(64_c, make_tuple(1_c, 4_c))))
                         .shape()),
            "(8,(4,16))");
  EXPECT_EQ(toString(size(make_composed_layout(
                Swz(3, 4, 3), 0_ct, make_layout(make_tuple(8_c, 64_d), make_tuple(64_d, 1_c))))),
            "?512{div=8}");
  EXPECT_EQ(toString(size(make_composed_layout(
                Swz(3, 4, 3), dyn(32, 32),
                make_layout(make_tuple(8_c, dyn(64, 8)), make_tuple(dyn(64, 8), 1_c))))),
            "?512{div=64}");
  EXPECT_EQ(toString(cosize(make_composed_layout(
                Swz(3, 4, 3), 0_ct,
                make_layout(make_tuple(8_c, dyn(64, 8)), make_tuple(dyn(64, 8), 1_c))))),
            "?512");
  EXPECT_EQ(rank(make_composed_layout(Swz(3, 4, 3), 0_ct,
                                      make_layout(make_tuple(8_c, make_tuple(4_c, 16_c)),
                                                  make_tuple(64_c, make_tuple(1_c, 4_c))))),
            2);
}

// The outer layout maps the coordinate, the offset is added and the inner function is applied
// last.
TEST(ComposedLayoutParity, Evaluation) {
  EXPECT_EQ(toString(crd2idx(
                make_tuple(5_c, 17_c),
                make_composed_layout(Swz(3, 4, 3), 0_ct,
                                     make_layout(make_tuple(8_c, 64_c), make_tuple(64_c, 1_c))))),
            "369");
  EXPECT_EQ(toString(crd2idx(337_dt, make_composed_layout(Swz(3, 4, 3), 0_ct,
                                                          make_layout(make_tuple(8_c, 64_c),
                                                                      make_tuple(64_c, 1_c))))),
            "?106");
  EXPECT_EQ(toString(crd2idx(
                make_tuple(5_d, 17_c),
                make_composed_layout(Swz(3, 4, 3), 0_ct,
                                     make_layout(make_tuple(8_c, 64_c), make_tuple(64_c, 1_c))))),
            "?369");
  EXPECT_EQ(toString(crd2idx(
                make_tuple(5_c, 17_c),
                make_composed_layout(Swz(3, 4, 3), 64_ct,
                                     make_layout(make_tuple(8_c, 64_c), make_tuple(64_c, 1_c))))),
            "417");
  EXPECT_EQ(toString(crd2idx(
                make_tuple(5_c, 17_c),
                make_composed_layout(Swz(3, 4, 3), 64_dt,
                                     make_layout(make_tuple(8_c, 64_c), make_tuple(64_c, 1_c))))),
            "?417");
  EXPECT_EQ(toString(crd2idx(
                5_ct, make_composed_layout(make_layout(make_tuple(4_c, 8_c), make_tuple(8_c, 1_c)),
                                           2_ct, make_layout(32_ct, 1_ct)))),
            "25");
  EXPECT_EQ(toString(crd2idx(
                make_tuple(dyn(4, 2), dyn(16, 8)),
                make_composed_layout(Swz(3, 4, 3), 0_ct,
                                     make_layout(make_tuple(8_c, 64_c), make_tuple(64_c, 1_c))))),
            "?304{div=8}");
  EXPECT_EQ(toString(crd2idx(
                make_tuple(2_c, 16_c),
                make_composed_layout(Swz(3, 4, 3), dyn(128, 16),
                                     make_layout(make_tuple(8_c, 64_c), make_tuple(64_c, 1_c))))),
            "?304{div=16}");
  EXPECT_EQ(toString(crd2idx(dyn(6, 2), make_composed_layout(
                                            make_layout(make_tuple(4_c, 8_c), make_tuple(8_c, 1_c)),
                                            2_ct, make_layout(32_ct, 1_ct)))),
            "?2");
}

// Layout algebra applies to the outer layout and keeps the inner function and offset.
TEST(ComposedLayoutParity, Algebra) {
  EXPECT_EQ(
      toString(get<1>(make_composed_layout(Swz(3, 4, 3), 0_ct,
                                           make_layout(make_tuple(8_c, make_tuple(4_c, 16_c)),
                                                       make_tuple(64_c, make_tuple(1_c, 4_c)))))),
      "S<3,4,3> o 0 o (4,16):(1,4)");
  EXPECT_EQ(toString(composition(
                make_composed_layout(Swz(3, 4, 3), 0_ct,
                                     make_layout(make_tuple(8_c, 64_c), make_tuple(64_c, 1_c))),
                make_layout(make_tuple(4_c, 8_c), make_tuple(1_c, 8_c)))),
            "S<3,4,3> o 0 o (4,8):(64,1)");
  EXPECT_EQ(toString(composition(
                make_composed_layout(Swz(3, 4, 3), 0_ct,
                                     make_layout(make_tuple(8_c, 64_c), make_tuple(64_c, 1_c))),
                make_tile(make_layout(4_ct, 2_ct), make_layout(16_ct, 1_ct)))),
            "S<3,4,3> o 0 o (4,16):(128,1)");
  EXPECT_EQ(toString(logical_divide(
                make_composed_layout(Swz(3, 4, 3), 0_ct,
                                     make_layout(make_tuple(8_c, 64_c), make_tuple(64_c, 1_c))),
                make_tile(make_layout(4_ct, 1_ct), make_layout(16_ct, 1_ct)))),
            "S<3,4,3> o 0 o ((4,2),(16,4)):((64,256),(1,16))");
  EXPECT_EQ(toString(zipped_divide(
                make_composed_layout(Swz(3, 4, 3), 0_ct,
                                     make_layout(make_tuple(8_c, 64_c), make_tuple(64_c, 1_c))),
                make_tile(make_layout(4_ct, 1_ct), make_layout(16_ct, 1_ct)))),
            "S<3,4,3> o 0 o ((4,16),(2,4)):((64,1),(256,16))");
  EXPECT_EQ(toString(composition(
                make_composed_layout(Swz(3, 4, 3), 0_ct,
                                     make_layout(make_tuple(8_c, 64_c), make_tuple(64_c, 1_c))),
                make_layout(make_tuple(dyn(4, 4), 8_c), make_tuple(1_c, 8_c)))),
            "S<3,4,3> o 0 o ((?4{div=4},?1),8):((64,1),1)");
  EXPECT_EQ(toString(logical_divide(
                make_composed_layout(Swz(3, 4, 3), 0_ct,
                                     make_layout(make_tuple(8_c, 64_c), make_tuple(64_c, 1_c))),
                make_tile(make_layout(4_ct, 1_ct), make_layout(dyn(16, 8), 1_ct)))),
            "S<3,4,3> o 0 o ((4,2),(?16{div=8},?4)):((64,256),(1,?16{div=8}))");
  EXPECT_EQ(toString(logical_divide(
                make_composed_layout(make_layout(256_ct, 1_ct), dyn(8, 8),
                                     make_layout(make_tuple(16_c, 16_c), make_tuple(16_c, 1_c))),
                make_tile(make_layout(4_ct, 1_ct), make_layout(dyn(8, 4), 1_ct)))),
            "256:1 o ?8{div=8} o ((4,4),(?8{div=4},?2)):((16,64),(1,?8{div=4}))");
  EXPECT_EQ(toString(tile_to_shape(
                make_composed_layout(Swz(3, 3, 3), 0_ct,
                                     make_layout(make_tuple(8_c, 64_c), make_tuple(64_c, 1_c))),
                make_tuple(128_c, 64_c))),
            "S<3,3,3> o 0 o ((8,16),(64,1)):((64,512),(1,0))");
  EXPECT_EQ(
      toString(coalesce(make_composed_layout(Swz(3, 4, 3), 0_ct,
                                             make_layout(make_tuple(8_c, make_tuple(4_c, 16_c)),
                                                         make_tuple(64_c, make_tuple(1_c, 4_c)))))),
      "S<3,4,3> o 0 o (8,64):(64,1)");
  EXPECT_EQ(toString(upcast(2, make_composed_layout(
                                   Swz(3, 4, 3), 0_ct,
                                   make_layout(make_tuple(8_c, 64_c), make_tuple(64_c, 1_c))))),
            "S<3,3,3> o 0 o (8,32):(32,1)");
  EXPECT_EQ(toString(downcast(2, make_composed_layout(
                                     Swz(3, 4, 3), 0_ct,
                                     make_layout(make_tuple(8_c, 64_c), make_tuple(64_c, 1_c))))),
            "S<3,5,3> o 0 o (8,128):(128,1)");
  // A recast scales the offset with the outer layout.
  EXPECT_EQ(toString(upcast(2, make_composed_layout(Swz(2, 3, 3), 4_ct, make_layout(32_ct, 1_ct)))),
            "S<2,2,3> o 2 o 16:1");
  EXPECT_EQ(
      toString(downcast(2, make_composed_layout(Swz(3, 4, 3), 3_ct, make_layout(64_ct, 1_ct)))),
      "S<3,5,3> o 6 o 128:1");
  // A slice moves its offset into the composition rather than the pointer.
  EXPECT_EQ(
      toString(slice(make_tuple(1_c, _), make_composed_layout(make_layout(128_ct, 2_ct), 1_ct,
                                                              make_layout(make_tuple(4_c, 3_c),
                                                                          make_tuple(1_c, 4_c))))),
      "128:2 o 2 o (3):(4)");
  EXPECT_EQ(toString(slice(
                make_tuple(_, dyn(2, 2)),
                make_composed_layout(make_layout(128_ct, 2_ct), 0_ct,
                                     make_layout(make_tuple(4_c, 3_c), make_tuple(1_c, 4_c))))),
            "128:2 o ?8{div=8} o (4):(1)");
  EXPECT_EQ(toString(slice(
                make_tuple(_, dyn(2, 2)),
                make_composed_layout(make_layout(128_ct, 2_ct), dyn(4, 4),
                                     make_layout(make_tuple(4_c, 3_c), make_tuple(1_c, 4_c))))),
            "128:2 o ?12{div=4} o (4):(1)");
  EXPECT_EQ(toString(slice_and_offset(make_tuple(1_d, _),
                                      make_composed_layout(
                                          make_layout(128_ct, 2_ct), 1_ct,
                                          make_layout(make_tuple(4_c, 3_c), make_tuple(1_c, 4_c))))
                         .first),
            "128:2 o ?2 o (3):(4)");
  EXPECT_EQ(toString(slice_and_offset(make_tuple(1_d, _),
                                      make_composed_layout(
                                          make_layout(128_ct, 2_ct), 1_ct,
                                          make_layout(make_tuple(4_c, 3_c), make_tuple(1_c, 4_c))))
                         .second),
            "0");
  // The complement is that of the outer layout.
  EXPECT_EQ(toString(complement(make_composed_layout(
                Swz(3, 4, 3), 0_ct, make_layout(make_tuple(8_c, 16_c), make_tuple(64_c, 1_c))))),
            "(4,1):(16,512)");
  EXPECT_EQ(
      toString(complement(
          make_composed_layout(make_layout(32_ct, 1_ct), 3_ct, make_layout(8_ct, 1_ct)), 32_ct)),
      "4:8");
  // With a swizzle inside, the alignment is the gcd of the swizzle, offset and outer alignments;
  // any other inner function gives 1.
  EXPECT_EQ(toString(max_alignment(make_composed_layout(
                Swz(3, 4, 3), 0_ct, make_layout(make_tuple(8_c, 64_c), make_tuple(64_c, 1_c))))),
            "16");
  EXPECT_EQ(toString(max_alignment(make_composed_layout(
                Swz(3, 4, 3), 8_ct, make_layout(make_tuple(8_c, 64_c), make_tuple(64_c, 1_c))))),
            "8");
  EXPECT_EQ(toString(max_alignment(
                make_composed_layout(make_layout(256_ct, 1_ct), 4_ct,
                                     make_layout(make_tuple(4_c, 3_c), make_tuple(1_c, 4_c))))),
            "1");
}

// Cases where Fly does not reproduce the reference; the expected results are the reference's.  Run
// with --gtest_also_run_disabled_tests to see them.
TEST(ComposedLayoutParity, DISABLED_KnownDivergences) {
  // CuTe C++ decays a swizzled slice that mixes no swizzled bits into a plain layout and moves the
  // offset to the pointer; Fly keeps S<3,4,3> o 5 o (8):(64).
  EXPECT_EQ(
      toString(slice(make_tuple(_, 5_c), make_composed_layout(Swz(3, 4, 3), 0_ct,
                                                              make_layout(make_tuple(8_c, 64_c),
                                                                          make_tuple(64_c, 1_c))))),
      "((2,4)):((64,144))");
  // Only CuTe C++ composes a layout with a composed layout (cutegen takes no composed rhs): it
  // composes with the inner layout and then the outer one; Fly keeps 32:2 o 0 o (4,2):(1,4).
  EXPECT_EQ(toString(composition(
                make_layout(64_ct, 2_ct),
                make_composed_layout(make_layout(32_ct, 1_ct), 0_ct,
                                     make_layout(make_tuple(4_c, 2_c), make_tuple(1_c, 4_c))))),
            "(4,2):(2,8)");
  // Only CuTe C++ composes a layout with a composed layout: it moves the layout across the
  // swizzle, which holds only when the layout is bitwise linear on the swizzle bits; Fly
  // rejects it as an unsupported swizzle composition.
  EXPECT_EQ(toString(composition(
                make_layout(64_ct, 2_ct),
                make_composed_layout(Swz(3, 4, 3), 0_ct,
                                     make_layout(make_tuple(8_c, 8_c), make_tuple(8_c, 1_c))))),
            "S<3,5,3> o 0 o (8,8):(16,2)");
}
