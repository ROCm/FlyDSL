//===- LayoutTest.cpp - Layout reference results ---------------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Layout algebra of Layout.hpp.
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

using namespace core_test;

// make_layout(shape) is column-major: static-1 modes get stride 0 and a dynamic shape makes the
// strides after it dynamic.
TEST(LayoutParity, Construction) {
  EXPECT_EQ(toString(make_layout(make_tuple(2_c, make_tuple(3_c, 4_c)))), "(2,(3,4)):(1,(2,6))");
  EXPECT_EQ(toString(make_layout(make_tuple(8_d, 2_c))), "(?8,2):(1,?8)");
  EXPECT_EQ(toString(make_layout(make_tuple(4_c, dyn(8, 4), 2_c))),
            "(4,?8{div=4},2):(1,4,?32{div=16})");
  EXPECT_EQ(toString(make_layout(make_tuple(1_c, 4_c, 1_c))), "(1,4,1):(0,1,0)");
  EXPECT_EQ(toString(make_layout(make_tuple(4_d, 8_c, make_tuple(2_c, 3_c)), LayoutRight{})),
            "(?4,8,(2,3)):(48,6,(3,1))");
  EXPECT_EQ(toString(make_layout(make_tuple(8_c))), "(8):(1)");
  EXPECT_EQ(toString(make_layout(make_layout(make_tuple(2_c, 3_c), make_tuple(1_c, 2_c)),
                                 make_layout(4_dt, 6_ct))),
            "((2,3),?4):((1,2),6)");
  EXPECT_EQ(toString(make_ordered_layout(make_tuple(2_c, 3_c, 4_c), make_tuple(2_c, 0_c, 1_c))),
            "(2,3,4):(12,1,3)");
  EXPECT_EQ(toString(make_ordered_layout(make_tuple(100_c, dyn(64, 8)), make_tuple(1_c, 0_c))),
            "(100,?64{div=8}):(?64{div=8},1)");
  EXPECT_EQ(
      toString(make_ordered_layout(make_tuple(2_c, 2_c, 2_c, 2_c), make_tuple(0_c, 2_c, 3_c, 1_c))),
      "(2,2,2,2):(1,4,8,2)");
  EXPECT_EQ(toString(make_ordered_layout(make_tuple(2_c, 3_c, 4_c, 5_c),
                                         make_tuple(2_c, 67_c, 42_c, 50_c))),
            "(2,3,4,5):(1,40,2,8)");
  // Dynamic order values rank after the static ones, left to right.
  EXPECT_EQ(toString(make_ordered_layout(make_tuple(2_c, 3_c, 4_c, 5_c),
                                         make_tuple(2_c, 67_d, 42_d, 50_c))),
            "(2,3,4,5):(1,10,30,2)");
  EXPECT_EQ(toString(make_layout_like(
                make_layout(make_tuple(2_c, 3_c, 4_c), make_tuple(1_c, 100_c, 20_c)))),
            "(2,3,4):(1,8,2)");
  EXPECT_EQ(toString(make_layout_like(
                make_layout(make_tuple(2_c, 3_c, 4_c, 5_c), make_tuple(0_c, 42_d, 1_c, 0_c)))),
            "(2,3,4,5):(0,4,1,0)");
  EXPECT_EQ(toString(make_layout_like(make_layout(make_tuple(4_c, 8_c), make_tuple(100_c, 1_c)))),
            "(4,8):(8,1)");
  EXPECT_EQ(toString(make_layout_like(make_layout(make_tuple(8_c, 4_c), make_tuple(0_c, 2_c)))),
            "(8,4):(0,1)");
  EXPECT_EQ(
      toString(make_layout_like(make_layout(make_tuple(8_c, 4_c, 6_c), make_tuple(1_c, 0_c, 2_c)))),
      "(8,4,6):(1,0,8)");
  EXPECT_EQ(toString(make_layout_like(
                make_layout(make_tuple(make_tuple(2_c, 2_c), make_tuple(2_c, 2_c)),
                            make_tuple(make_tuple(1_c, 4_c), make_tuple(8_c, 32_c))))),
            "((2,2),(2,2)):((1,2),(4,8))");
  EXPECT_EQ(toString(make_layout_like(make_layout(make_tuple(2_c, make_tuple(3_c, 4_c)),
                                                  make_tuple(50_c, make_tuple(1_c, 8_c))))),
            "(2,(3,4)):(12,(1,3))");
  EXPECT_EQ(toString(make_layout_like(
                make_layout(make_tuple(2_c, 2_c, 2_c, 2_c), make_tuple(0_c, 2_c, 4_c, 1_c)))),
            "(2,2,2,2):(0,2,4,1)");
  EXPECT_EQ(
      toString(make_fragment_like(make_layout(make_tuple(make_tuple(2_c, 2_c), 4_c, 2_c),
                                              make_tuple(make_tuple(16_c, 7_c), 128_c, 1_c)))),
      "((2,2),4,2):((1,2),8,4)");
  // Sources differ: CuTe C++ keeps the rank-1 rest group whole; Fly unpacks it, as here.
  EXPECT_EQ(toString(make_fragment_like(make_layout(make_tuple(4_c, 3_c), make_tuple(3_c, 1_c)))),
            "(4,3):(1,4)");
  EXPECT_EQ(toString(make_fragment_like(make_layout(make_tuple(2_c, 3_d), make_tuple(3_c, 1_c)))),
            "(2,?3):(1,2)");
  EXPECT_EQ(toString(make_fragment_like(make_layout(8_ct, 2_ct))), "8:1");
  EXPECT_EQ(toString(make_fragment_like(
                make_layout(make_tuple(2_c, 3_c, 4_c), make_tuple(1_c, 2_c, 6_c)))),
            "(2,3,4):(1,2,6)");
  EXPECT_EQ(toString(make_fragment_like(
                make_layout(make_tuple(2_c, 3_c, 4_c), make_tuple(12_c, 4_c, 1_c)))),
            "(2,3,4):(1,8,2)");
  // Sources differ: CuTe C++ keeps the rank-1 rest group whole; Fly unpacks it, as here.
  EXPECT_EQ(toString(make_fragment_like(make_layout(make_tuple(make_tuple(2_c, 2_c), 3_c),
                                                    make_tuple(make_tuple(16_c, 7_c), 32_c)))),
            "((2,2),3):((1,2),4)");
  EXPECT_EQ(toString(make_identity_layout(make_tuple(4_c, 3_d))), "(4,?3):(1E0,1E1)");
  EXPECT_EQ(toString(make_layout(make_tuple(dyn(8, 2), make_tuple(4_c, dyn(6, 3))))),
            "(?8{div=2},(4,?6{div=3})):(1,(?8{div=2},?32{div=8}))");
}

TEST(LayoutParity, SizeAndCosize) {
  EXPECT_EQ(toString(size(make_layout(make_tuple(make_tuple(2_c, 3_d), 4_c),
                                      make_tuple(make_tuple(1_c, 2_c), 6_d)))),
            "?24{div=8}");
  EXPECT_EQ(toString(cosize(make_layout(make_tuple(4_c, 8_c), make_tuple(1_c, 8_c)))), "60");
  EXPECT_EQ(toString(cosize(make_layout(make_tuple(4_c, 8_d), make_tuple(1_c, 16_d)))), "?116");
  EXPECT_EQ(
      toString(coshape(make_layout(make_tuple(4_c, 3_c), make_tuple(E(1_ct, {0}), E(2_ct, {1}))))),
      "(4,5)");
}

// A layout maps a coordinate to an index; a static zero stride removes a dynamic coordinate.
TEST(LayoutParity, Evaluation) {
  EXPECT_EQ(toString(crd2idx(make_tuple(make_tuple(1_c, 2_c), 3_c),
                             make_layout(make_tuple(make_tuple(2_c, 3_c), 4_c),
                                         make_tuple(make_tuple(1_c, 2_c), 6_c)))),
            "23");
  EXPECT_EQ(toString(crd2idx(17_dt, make_layout(make_tuple(make_tuple(2_c, 3_c), 4_c),
                                                make_tuple(make_tuple(1_c, 2_c), 6_c)))),
            "?17");
  EXPECT_EQ(toString(crd2idx(make_tuple(1_c, 2_c),
                             make_layout(make_tuple(4_c, 8_c), make_tuple(32_d, 1_c)))),
            "?34");
  EXPECT_EQ(toString(crd2idx(make_tuple(3_d, 2_c),
                             make_layout(make_tuple(4_c, 8_c), make_tuple(0_c, 1_c)))),
            "2");
  EXPECT_EQ(
      toString(crd2idx(make_tuple(3_d, 2_c),
                       make_layout(make_tuple(4_c, 8_c), make_tuple(E(1_ct, {1}), E(1_ct, {0}))))),
      "(2,?3)");
  EXPECT_EQ(toString(crd2idx(dyn(16, 8), make_layout(make_tuple(4_c, 8_c), make_tuple(8_c, 1_c)))),
            "?4{div=2}");
}

// get_hier_coord inverts the layout into a coordinate of its shape; get_flat_coord flattens each
// mode of it and get_1d_coord the whole of it.
TEST(LayoutParity, Coordinates) {
  EXPECT_EQ(
      toString(
          make_layout(make_tuple(2_c, 3_c, 4_c), make_tuple(1_c, 2_c, 6_c)).get_hier_coord(23_ct)),
      "(1,2,3)");
  EXPECT_EQ(toString(make_layout(make_tuple(make_tuple(2_c, 3_c), 4_c),
                                 make_tuple(make_tuple(1_c, 2_c), 6_c))
                         .get_hier_coord(17_ct)),
            "((1,2),2)");
  EXPECT_EQ(
      toString(
          make_layout(make_tuple(2_c, 3_c, 4_c), make_tuple(1_c, 2_c, 6_c)).get_hier_coord(23_dt)),
      "(?1,?2,?3)");
  EXPECT_EQ(toString(make_layout(make_tuple(2_c, 3_c), make_tuple(3_c, 1_c)).get_hier_coord(4_ct)),
            "(1,1)");
  EXPECT_EQ(toString(make_layout(make_tuple(make_tuple(2_c, 4_c), 8_c),
                                 make_tuple(make_tuple(1_c, 2_c), 8_c))
                         .get_flat_coord(5_ct)),
            "(5,0)");
  EXPECT_EQ(toString(make_layout(make_tuple(make_tuple(2_c, 4_c), 8_c),
                                 make_tuple(make_tuple(1_c, 2_c), 8_c))
                         .get_flat_coord(13_dt)),
            "(?5,?1)");
  EXPECT_EQ(
      toString(
          make_layout(make_tuple(2_c, 3_c, 4_c), make_tuple(1_c, 2_c, 6_c)).get_1d_coord(23_ct)),
      "23");
  EXPECT_EQ(toString(make_layout(make_tuple(2_c, 3_c), make_tuple(3_c, 1_c)).get_1d_coord(4_ct)),
            "3");
  EXPECT_EQ(
      toString(make_layout(make_tuple(4_c, dyn(8, 4)), make_tuple(1_c, 4_c)).get_hier_coord(13_ct)),
      "(1,?3)");
}

TEST(LayoutParity, ModeAccess) {
  EXPECT_EQ(toString(get<0>(make_layout(make_tuple(make_tuple(2_c, 3_c), 4_d),
                                        make_tuple(make_tuple(1_c, 2_c), 6_c)))),
            "(2,3):(1,2)");
  EXPECT_EQ(toString(take<1, 3>(
                make_layout(make_tuple(2_c, 3_d, 4_c, 5_c), make_tuple(1_c, 2_c, 6_d, 24_c)))),
            "(?3,4):(2,?6)");
  EXPECT_EQ(
      toString(select<2, 0>(make_layout(make_tuple(2_c, 3_d, 4_c), make_tuple(1_c, 2_c, 6_d)))),
      "(4,2):(?6,1)");
  EXPECT_EQ(toString(group<1, 3>(
                make_layout(make_tuple(2_c, 3_d, 4_c, 5_c), make_tuple(1_c, 2_c, 6_d, 24_c)))),
            "(2,(?3,4),5):(1,(2,?6),24)");
  EXPECT_EQ(toString(flatten(make_layout(
                make_tuple(make_tuple(2_c, 3_d), make_tuple(4_c, make_tuple(5_c, 6_c))),
                make_tuple(make_tuple(1_c, 2_c), make_tuple(6_d, make_tuple(24_c, 120_c)))))),
            "(2,?3,4,5,6):(1,2,?6,24,120)");
  EXPECT_EQ(toString(append(make_layout(make_tuple(2_c, 3_c), make_tuple(1_c, 2_c)),
                            make_layout(4_dt, 6_ct))),
            "(2,3,?4):(1,2,6)");
  EXPECT_EQ(toString(prepend(make_layout(make_tuple(2_c, 3_c), make_tuple(1_c, 2_c)),
                             make_layout(4_dt, 6_ct))),
            "(?4,2,3):(6,1,2)");
  EXPECT_EQ(toString(slice(make_tuple(_, 1_c, _),
                           make_layout(make_tuple(2_c, 3_d, make_tuple(4_c, 5_c)),
                                       make_tuple(1_c, 2_c, make_tuple(6_d, 24_c))))),
            "(2,(4,5)):(1,(?6,24))");
  EXPECT_EQ(toString(dice(make_tuple(_, 1_c, _),
                          make_layout(make_tuple(2_c, 3_d, make_tuple(4_c, 5_c)),
                                      make_tuple(1_c, 2_c, make_tuple(6_d, 24_c))))),
            "(?3):(2)");
}

// Static modes merge when the stride chains; dynamic ones never do.  A static-1 mode disappears
// even with a dynamic stride.
TEST(LayoutParity, Coalesce) {
  EXPECT_EQ(toString(coalesce(make_layout(make_tuple(2_c, make_tuple(1_c, 6_c)),
                                          make_tuple(1_c, make_tuple(6_d, 2_c))))),
            "12:1");
  EXPECT_EQ(toString(coalesce(make_layout(
                make_tuple(make_tuple(2_c, make_tuple(3_c, 4_c)), make_tuple(3_c, 2_c), 1_c),
                make_tuple(make_tuple(4_c, make_tuple(8_c, 24_c)), make_tuple(2_c, 6_c), 12_c)))),
            "(24,6):(4,2)");
  EXPECT_EQ(toString(coalesce(make_layout(make_tuple(2_c, 1_c, 6_c), make_tuple(1_c, 6_c, 2_c)))),
            "12:1");
  EXPECT_EQ(toString(coalesce(make_layout(make_tuple(2_c, 4_d, 3_c), make_tuple(1_c, 2_c, 8_c)))),
            "(2,?4,3):(1,2,8)");
  EXPECT_EQ(toString(coalesce(make_layout(make_tuple(2_c, 4_c, 3_c), make_tuple(1_c, 2_d, 8_c)))),
            "(2,4,3):(1,?2,8)");
  EXPECT_EQ(toString(coalesce(make_layout(make_tuple(2_c, 1_d, 6_c), make_tuple(1_c, 7_d, 2_c)))),
            "(2,?1,6):(1,?7,2)");
  EXPECT_EQ(toString(coalesce(make_layout(make_tuple(2_c, dyn(4, 2), 3_c, 5_c),
                                          make_tuple(1_c, 2_c, dyn(8, 8), dyn(24, 8))))),
            "(2,?4{div=2},3,5):(1,2,?8{div=8},?24{div=8})");
  EXPECT_EQ(toString(coalesce(make_layout(make_tuple(2_c, 1_c, 6_c), make_tuple(4_c, 7_c, 8_c)))),
            "12:4");
  EXPECT_EQ(
      toString(coalesce(make_layout(make_tuple(make_tuple(2_c, 2_c), make_tuple(2_c, 2_c)),
                                    make_tuple(make_tuple(1_c, 4_c), make_tuple(8_c, 32_c))))),
      "(2,4,2):(1,4,32)");
  EXPECT_EQ(toString(coalesce(make_layout(make_tuple(2_c, 4_c, 6_c), make_tuple(24_c, 6_c, 1_c)))),
            "(2,4,6):(24,6,1)");
  // Sources differ: CuTe C++ canonicalizes an all-size-1 layout to 1:0
  EXPECT_EQ(toString(coalesce(make_layout(make_tuple(1_c, 1_c), make_tuple(5_c, 7_c)))), "1:7");
  EXPECT_EQ(toString(coalesce(make_layout(make_tuple(2_c, make_tuple(1_c, 6_c)),
                                          make_tuple(1_c, make_tuple(6_c, 2_c))),
                              make_tuple(1_c, 1_c))),
            "(2,6):(1,2)");
  // The profile coalesces the modes it covers and leaves the others.
  EXPECT_EQ(toString(coalesce(make_layout(make_tuple(make_tuple(2_c, 3_c), 4_c),
                                          make_tuple(make_tuple(1_c, 2_c), 6_c)),
                              make_tuple(1_c))),
            "(6,4):(1,6)");
  EXPECT_EQ(toString(coalesce(
                make_layout(make_tuple(make_tuple(16_c, 4_c), 128_c),
                            make_tuple(make_tuple(E(1_ct, {0}), E(16_ct, {0})), E(1_ct, {1}))))),
            "(64,128):(1E0,1E1)");
  EXPECT_EQ(
      toString(coalesce(make_layout(make_tuple(2_c, 4_c), make_tuple(E(1_ct, {1}), E(2_ct, {1}))))),
      "8:1E1");
  EXPECT_EQ(
      toString(coalesce(make_layout(make_tuple(2_c, 4_c), make_tuple(E(1_ct, {0}), E(1_ct, {1}))))),
      "(2,4):(1E0,1E1)");
  EXPECT_EQ(toString(coalesce(make_layout(make_tuple(make_tuple(2_c, 2_c), make_tuple(2_c, 2_c)),
                                          make_tuple(make_tuple(E(1_ct, {0}), E(1_ct, {1})),
                                                     make_tuple(E(2_ct, {1}), E(2_ct, {0})))))),
            "(2,4,2):(1E0,1E1,2E0)");
  EXPECT_EQ(toString(filter(make_layout(make_tuple(4_c, make_tuple(2_c, 3_c)),
                                        make_tuple(0_c, make_tuple(1_c, 0_c))))),
            "2:1");
  EXPECT_EQ(toString(filter(make_layout(make_tuple(4_c, 1_c), make_tuple(0_c, 3_c)))), "1:0");
  EXPECT_EQ(toString(filter_zeros(make_layout(make_tuple(4_c, make_tuple(2_c, 3_c)),
                                              make_tuple(0_c, make_tuple(1_c, 0_c))))),
            "(1,(2,1)):(0,(1,0))");
}

// Each mode of the result is static when the lhs strides and shapes it consumes are; the stride of
// a composed mode is the product of a rhs stride and a lhs stride, so their divisibilities
// multiply.
TEST(LayoutParity, Composition) {
  EXPECT_EQ(toString(composition(make_layout(make_tuple(10_c, 2_c), make_tuple(16_c, 4_c)),
                                 make_layout(make_tuple(5_c, 4_c), make_tuple(1_c, 5_c)))),
            "(5,(2,2)):(16,(80,4))");
  // A dynamic size-1 mode cannot be dropped.
  EXPECT_EQ(toString(composition(make_layout(make_tuple(10_d, 2_d), make_tuple(16_d, 4_d)),
                                 make_layout(make_tuple(5_d, 4_d), make_tuple(1_d, 5_d)))),
            "((?5,?1),(?2,?2)):((?16,?4),(?80,?4))");
  EXPECT_EQ(toString(composition(make_layout(make_tuple(6_c, 2_c), make_tuple(8_d, 2_c)),
                                 make_layout(make_tuple(4_c, 3_c), make_tuple(3_c, 1_c)))),
            "((2,2),3):((?24{div=3},2),?8)");
  EXPECT_EQ(toString(composition(make_layout(make_tuple(12_d, make_tuple(4_d, 8_d)),
                                             make_tuple(59_d, make_tuple(13_d, 1_d))),
                                 make_tile(make_layout(3_ct, 1_ct), make_layout(8_ct, 1_ct)))),
            "(3,(?4,?2)):(?59,(?13,?1))");
  EXPECT_EQ(toString(composition(make_layout(20_ct, 2_ct),
                                 make_layout(make_tuple(5_c, 4_c), make_tuple(4_c, 1_c)))),
            "(5,4):(8,2)");
  EXPECT_EQ(toString(composition(make_layout(make_tuple(6_c, 2_c), make_tuple(8_c, 2_c)),
                                 make_layout(make_tuple(4_c, 3_c), make_tuple(3_c, 1_c)))),
            "((2,2),3):((24,2),8)");
  // The rhs splits at the lhs mode boundaries, keeps its own hierarchy and its zero strides.
  EXPECT_EQ(toString(composition(make_layout(make_tuple(4_c, 3_c), make_tuple(2_c, 10_c)),
                                 make_layout(8_ct, 1_ct))),
            "(4,2):(2,10)");
  EXPECT_EQ(toString(composition(make_layout(make_tuple(4_c, 3_c), make_tuple(2_c, 10_c)),
                                 make_layout(make_tuple(make_tuple(2_c, 2_c), 2_c),
                                             make_tuple(make_tuple(1_c, 2_c), 4_c)))),
            "((2,2),2):((2,4),10)");
  EXPECT_EQ(toString(composition(make_layout(make_tuple(4_c, 3_c), make_tuple(2_c, 10_c)),
                                 make_layout(make_tuple(5_c, 2_c), make_tuple(0_c, 4_c)))),
            "(5,2):(0,10)");
  EXPECT_EQ(toString(composition(make_layout(make_tuple(4_c, 2_c), make_tuple(1_c, 5_c)),
                                 make_layout(8_ct, -2_ct))),
            "(2,4):(-2,-5)");
  // The result is not coalesced.
  EXPECT_EQ(toString(composition(make_layout(make_tuple(4_c, 8_d), make_tuple(1_c, 4_c)),
                                 make_layout(16_ct, 1_ct))),
            "(4,4):(1,4)");
  EXPECT_EQ(
      toString(composition(make_layout(make_tuple(make_tuple(4_c, 3_c), make_tuple(2_c, 5_c)),
                                       make_tuple(make_tuple(15_c, 1_c), make_tuple(3_c, 60_c))),
                           make_layout(make_tuple(4_c, make_tuple(3_c, 2_c)),
                                       make_tuple(6_c, make_tuple(1_c, 24_c))))),
      "!incompatible-shape-division:!incompatible-shape-division");
  EXPECT_EQ(toString(composition(make_layout(make_tuple(4_c, 8_c), make_tuple(1_c, 4_c)),
                                 make_layout(6_dt, 2_dt))),
            "?6:?2");
  // A dynamic shape keeps the size-1 modes it may produce.
  EXPECT_EQ(
      toString(composition(make_layout(make_tuple(dyn(16, 4), 8_c), make_tuple(1_c, dyn(16, 4))),
                           make_layout(make_tuple(4_c, 8_c), make_tuple(1_c, dyn(16, 4))))),
      "((?4{div=4},?1),(?1,?8)):((1,?16{div=4}),(?16{div=4},?16{div=4}))");
  EXPECT_EQ(
      toString(composition(
          make_layout(make_tuple(4_c, make_tuple(2_c, 3_c)), make_tuple(6_c, make_tuple(3_c, 1_c))),
          make_layout(make_tuple(2_c, 4_c), make_tuple(E(1_ct, {1, 1}), E(1_ct, {0}))))),
      "(2,4):(1,6)");
  EXPECT_EQ(toString(composition(
                make_layout(make_tuple(4_c, 4_c), make_tuple(4_c, 1_c)),
                make_layout(make_tuple(4_c, 4_c), make_tuple(E(1_ct, {1}), E(1_ct, {0}))))),
            "(4,4):(1,4)");
  // A basis stride selects from the lhs before its size-1 modes are dropped.
  EXPECT_EQ(toString(composition(
                make_layout(make_tuple(make_tuple(1_c), 8_c), make_tuple(make_tuple(3_c), 4_c)),
                make_layout(2_ct, E(1_ct, {0})))),
            "2:3");
  EXPECT_EQ(toString(composition(make_layout(make_tuple(2_c, 3_c), make_tuple(1_c, 2_c)),
                                 make_layout(3_ct, E(1_ct, {1})))),
            "3:2");
  EXPECT_EQ(toString(composition(make_layout(make_tuple(4_c, 3_c), make_tuple(1_c, 8_c)),
                                 make_layout(3_ct, 5_ct))),
            "!incompatible-shape-division:!incompatible-shape-division");
  EXPECT_EQ(toString(composition(make_layout(make_tuple(4_c, 8_c), make_tuple(1_c, 4_c)),
                                 make_layout(dyn(16, 4), 2_ct))),
            "?16{div=4}:2");
  EXPECT_EQ(toString(composition(make_layout(16_ct, 2_ct), make_layout(dyn(8, 4), dyn(2, 2)))),
            "?8{div=4}:?4{div=4}");
  EXPECT_EQ(toString(composition(make_layout(dyn(32, 8), dyn(4, 2)),
                                 make_layout(make_tuple(4_c, 2_c), make_tuple(2_c, 8_c)))),
            "(4,2):(?8{div=4},?32{div=16})");
  EXPECT_EQ(toString(composition(
                make_layout(make_tuple(dyn(64, 16), dyn(32, 8)), make_tuple(1_c, dyn(64, 16))),
                make_tile(make_layout(16_ct, 2_ct), make_layout(8_ct, 1_ct)))),
            "(16,8):(2,?64{div=16})");
}

// Without a cotarget the complement ends in a size-1 mode carrying the next stride.
TEST(LayoutParity, Complement) {
  EXPECT_EQ(toString(complement(make_layout(2_ct, 3_ct))), "(3,1):(1,6)");
  EXPECT_EQ(toString(complement(make_layout(2_ct, 3_ct), 12_ct)), "(3,2):(1,6)");
  EXPECT_EQ(toString(complement(make_layout(2_ct, 3_ct), 6_ct)), "3:1");
  EXPECT_EQ(toString(complement(make_layout(4_ct, 2_ct))), "(2,1):(1,8)");
  EXPECT_EQ(toString(complement(make_layout(1_ct, 0_ct))), "1:1");
  EXPECT_EQ(toString(complement(make_layout(make_tuple(2_c, 3_c), make_tuple(1_c, 0_c)))), "1:2");
  EXPECT_EQ(toString(complement(make_layout(make_tuple(4_c, 8_c), make_tuple(1_c, 4_c)))), "1:32");
  EXPECT_EQ(toString(complement(make_layout(make_tuple(2_c, make_tuple(3_c, 4_c)),
                                            make_tuple(1_c, make_tuple(6_c, 36_c))),
                                288_ct)),
            "(3,2,2):(2,18,144)");
  EXPECT_EQ(toString(complement(make_layout(4_ct, 2_ct), 64_dt)), "(2,?8):(1,8)");
  EXPECT_EQ(toString(complement(make_layout(4_ct, 0_ct), 8_ct)), "8:1");
  EXPECT_EQ(toString(complement(make_layout(
                make_tuple(2_c, 5_c, 3_c), make_tuple(E(4_ct, {1}), E(5_ct, {0}), E(16_ct, {1}))))),
            "((5,1),(4,2,1)):((1E0,25E0),(1E1,8E1,48E1))");
  EXPECT_EQ(toString(complement(make_layout(3_ct, E(1_ct, {0})))), "(1):(3E0)");
  EXPECT_EQ(toString(complement(make_layout(3_ct, E(4_ct, {2})))),
            "(1,1,(4,1)):(1E0,1E1,(1E2,12E2))");
  EXPECT_EQ(
      toString(complement(make_layout(make_tuple(2_c, 3_c), make_tuple(E(1_ct, {0}), E(1_ct, {1}))),
                          make_tuple(8_c, 9_c))),
      "(4,3):(2E0,3E1)");
  EXPECT_EQ(toString(complement(make_layout(make_tuple(4_c, 2_c), make_tuple(1_c, 2_c)), 8_ct)),
            "!non-injective-layout:!non-injective-layout");
  EXPECT_EQ(toString(complement(make_layout(4_ct, dyn(2, 2)), 64_ct)),
            "(?2{div=2},?8):(1,?8{div=8})");
  // A dynamic stride cannot be ordered against the other strides.
  EXPECT_EQ(toString(complement(make_layout(make_tuple(4_c, 8_c), make_tuple(1_c, dyn(8, 8))),
                                dyn(256, 64))),
            "!unsupported-dynamic-complement:!unsupported-dynamic-complement");
}

TEST(LayoutParity, Inverse) {
  EXPECT_EQ(toString(right_inverse(make_layout(make_tuple(2_c, 4_c), make_tuple(4_c, 1_c)))),
            "(4,2):(2,1)");
  EXPECT_EQ(toString(right_inverse(
                make_layout(make_tuple(make_tuple(3_c, 2_c), make_tuple(4_c, 5_c)),
                            make_tuple(make_tuple(5_c, 60_c), make_tuple(15_c, 1_c))))),
            "(5,3,4,2):(24,1,6,3)");
  EXPECT_EQ(toString(right_inverse(make_layout(make_tuple(4_c, 2_d), make_tuple(1_c, 4_c)))),
            "(4,?2):(1,4)");
  EXPECT_EQ(toString(right_inverse(make_layout(make_tuple(4_c, 2_c), make_tuple(1_c, 9_d)))),
            "4:1");
  EXPECT_EQ(toString(left_inverse(make_layout(make_tuple(4_c, 2_c), make_tuple(2_c, 1_c)))),
            "(2,4):(4,1)");
  EXPECT_EQ(toString(left_inverse(make_layout(make_tuple(4_c, 2_c), make_tuple(1_c, 8_c)))),
            "(8,2):(1,4)");
  EXPECT_EQ(toString(right_inverse(make_layout(make_tuple(4_c, 1_c, 5_c),
                                               make_tuple(E(1_ct, {0}), 0_c, E(1_ct, {1}))))),
            "(4,5):(1,4)");
  EXPECT_EQ(toString(right_inverse(make_layout(make_tuple(2_c, 3_c), make_tuple(1_c, 2_c)))),
            "6:1");
  EXPECT_EQ(toString(right_inverse(
                make_layout(make_tuple(4_c, 5_c), make_tuple(E(1_ct, {1}), E(1_ct, {0}))))),
            "(5,4):(4,1)");
  EXPECT_EQ(toString(left_inverse(
                make_layout(make_tuple(4_c, 5_c), make_tuple(E(1_ct, {1}), E(1_ct, {0}))))),
            "(5,4):(4,1)");
  EXPECT_EQ(toString(right_inverse(
                make_layout(make_tuple(4_c, 5_c), make_tuple(E(2_ct, {0}), E(2_ct, {1}))))),
            "(1,1):(0,0)");
  EXPECT_EQ(toString(left_inverse(
                make_layout(make_tuple(4_c, 5_c), make_tuple(E(2_ct, {0}), E(2_ct, {1}))))),
            "((2,4),(2,5)):((0,1),(0,4))");
  EXPECT_EQ(toString(right_inverse(
                make_layout(make_tuple(make_tuple(2_c, 2_c, 2_c), make_tuple(2_c, 2_c, 2_c)),
                            make_tuple(make_tuple(E(1_ct, {0}), E(2_ct, {1}), E(4_ct, {0})),
                                       make_tuple(E(1_ct, {1}), E(2_ct, {0}), E(4_ct, {1})))))),
            "((2,2,2),(2,2,2)):((1,16,4),(8,2,32))");
  // A gap ends the right inverse of its own axis only.
  EXPECT_EQ(toString(right_inverse(
                make_layout(make_tuple(make_tuple(2_c, 2_c, 2_c), make_tuple(2_c, 2_c, 2_c)),
                            make_tuple(make_tuple(E(1_ct, {0}), E(2_ct, {1}), E(5_ct, {0})),
                                       make_tuple(E(1_ct, {1}), E(2_ct, {0}), E(4_ct, {1})))))),
            "((2,2),(2,2,2)):((1,16),(8,2,32))");
  EXPECT_EQ(toString(left_inverse(
                make_layout(make_tuple(make_tuple(2_c, 2_c, 2_c), make_tuple(2_c, 2_c, 2_c)),
                            make_tuple(make_tuple(E(1_ct, {0}), E(2_ct, {1}), E(6_ct, {0})),
                                       make_tuple(E(1_ct, {1}), E(2_ct, {0}), E(6_ct, {1})))))),
            "((2,3,2),(2,3,2)):((1,16,4),(8,2,32))");
  EXPECT_EQ(toString(left_inverse(make_layout(make_tuple(4_c, 2_c), make_tuple(2_c, 3_c)))),
            "!non-injective-layout:!non-injective-layout");
}

// The tile mode is composition with the tiler; the rest mode is its complement bounded by the
// coalesced shape, so a dynamic shape yields a dynamic rest.
TEST(LayoutParity, LogicalDivide) {
  EXPECT_EQ(
      toString(logical_divide(make_layout(make_tuple(4_c, 2_c, 3_c), make_tuple(2_c, 1_c, 8_c)),
                              make_layout(4_ct, 2_ct))),
      "((2,2),(2,3)):((4,1),(2,8))");
  EXPECT_EQ(
      toString(logical_divide(make_layout(make_tuple(9_c, make_tuple(4_c, 8_c)),
                                          make_tuple(59_c, make_tuple(13_c, 1_c))),
                              make_tile(make_layout(3_ct, 3_ct),
                                        make_layout(make_tuple(2_c, 4_c), make_tuple(1_c, 8_c))))),
      "((3,3),((2,4),(2,2))):((177,59),((13,2),(26,1)))");
  EXPECT_EQ(toString(logical_divide(make_layout(make_tuple(6_c, 6_c), make_tuple(1_c, 12_c)),
                                    make_layout(make_tuple(6_c, 3_c), make_tuple(3_c, 1_c)))),
            "(((2,3),3),2):(((3,12),1),36)");
  EXPECT_EQ(toString(logical_divide(make_layout(1_ct, 1_ct), make_layout(1_ct, 1_ct))),
            "(1,1):(1,0)");
  EXPECT_EQ(toString(logical_divide(make_layout(6_ct, 1_ct), make_layout(2_ct, 1_ct))),
            "(2,3):(1,2)");
  EXPECT_EQ(toString(logical_divide(make_layout(6_ct, 1_ct), make_layout(2_ct, 3_ct))),
            "(2,3):(3,1)");
  EXPECT_EQ(toString(logical_divide(make_layout(6_ct, 1_ct),
                                    make_layout(make_tuple(2_c, 3_c), make_tuple(3_c, 1_c)))),
            "((2,3),1):((3,1),0)");
  EXPECT_EQ(toString(logical_divide(make_layout(6_ct, 2_ct), make_layout(2_ct, 3_ct))),
            "(2,3):(6,2)");
  EXPECT_EQ(toString(logical_divide(make_layout(make_tuple(6_c, 6_c), make_tuple(12_c, 1_c)),
                                    make_layout(make_tuple(6_c, 3_c), make_tuple(3_c, 1_c)))),
            "(((2,3),3),2):(((36,1),12),3)");
  EXPECT_EQ(toString(logical_divide(make_layout(32_ct, 1_ct), make_layout(2_ct, 8_ct))),
            "(2,(8,2)):(8,(1,16))");
  EXPECT_EQ(toString(logical_divide(make_layout(make_tuple(4_c, 1_c), make_tuple(1_c, 1_c)),
                                    make_layout(2_ct, 2_ct))),
            "(2,2):(2,1)");
  EXPECT_EQ(toString(logical_divide(make_layout(make_tuple(8_c, 8_c), make_tuple(8_c, 1_c)),
                                    make_layout(32_ct, 2_ct))),
            "((4,8),2):((16,1),8)");
  EXPECT_EQ(toString(logical_divide(make_layout(4_ct, 1_ct), make_layout(1_ct, 3_ct))),
            "(1,4):(3,1)");
  // The complement is bounded by the coalesced shape 15, not by the mode-wise (3,5).
  EXPECT_EQ(toString(logical_divide(make_layout(make_tuple(3_c, 5_c), make_tuple(1_c, 3_c)),
                                    make_layout(2_ct, 1_ct))),
            "(2,8):(1,2)");
  EXPECT_EQ(toString(logical_divide(make_layout(make_tuple(6_c, 6_d), make_tuple(1_c, 6_c)),
                                    make_layout(2_ct, 3_ct))),
            "(2,(3,?6)):(3,(1,6))");
  EXPECT_EQ(toString(logical_divide(make_layout(48_dt, 1_ct), make_layout(32_ct, 1_ct))),
            "(32,?2):(1,32)");
  EXPECT_EQ(toString(logical_divide(make_layout(dyn(64, 16), 1_ct), make_layout(4_ct, 1_ct))),
            "(4,?16{div=4}):(1,4)");
  EXPECT_EQ(
      toString(logical_divide(make_layout(make_tuple(8_c, dyn(32, 8)), make_tuple(dyn(32, 8), 1_c)),
                              make_tile(make_layout(2_ct, 1_ct), make_layout(4_ct, 1_ct)))),
      "((2,4),(4,?8{div=2})):((?32{div=8},?64{div=16}),(1,4))");
  EXPECT_EQ(
      toString(logical_divide(make_layout(make_tuple(32_c, 4_c, 4_d), make_tuple(1_c, 32_c, 128_c)),
                              make_layout(64_ct, 1_ct))),
      "(64,(2,?4)):(1,(64,128))");
  EXPECT_EQ(toString(logical_divide(make_layout(make_tuple(8_c, 16_d), make_tuple(16_d, 1_c)),
                                    make_tile(_, make_layout(4_ct, 1_ct)))),
            "(8,(4,?4)):(?16,(1,4))");
  EXPECT_EQ(toString(logical_divide(
                make_layout(make_tuple(8_c, 16_d), make_tuple(E(1_ct, {0}), E(1_ct, {1}))),
                make_tile(make_layout(4_ct, 1_ct), make_layout(4_ct, 1_ct)))),
            "((4,2),(4,?4)):((1E0,4E0),(1E1,4E1))");
  // A basis stride in the tiler selects the layout mode it divides.
  EXPECT_EQ(toString(logical_divide(make_layout(make_tuple(8_c, 8_c), make_tuple(9_c, 1_c)),
                                    make_layout(4_ct, E(1_ct, {0})))),
            "(4,(2,8)):(9,(36,1))");
  EXPECT_EQ(toString(logical_divide(make_layout(make_tuple(8_c, 8_c), make_tuple(9_c, 1_c)),
                                    make_layout(4_ct, E(1_ct, {1})))),
            "(4,(8,2)):(1,(9,4))");
  EXPECT_EQ(toString(logical_divide(make_layout(make_tuple(8_c, 8_c), make_tuple(9_c, 1_c)),
                                    make_layout(4_ct, E(2_ct, {0})))),
            "(4,(2,8)):(18,(9,1))");
  EXPECT_EQ(toString(logical_divide(
                make_layout(make_tuple(8_c, 8_c), make_tuple(9_c, 1_c)),
                make_layout(make_tuple(4_c, 4_c), make_tuple(E(1_ct, {1}), E(2_ct, {0}))))),
            "((4,4),(2,2)):((1,18),(9,4))");
  EXPECT_EQ(
      toString(zipped_divide(make_layout(make_tuple(9_c, make_tuple(4_c, 8_c)),
                                         make_tuple(59_c, make_tuple(13_c, 1_c))),
                             make_tile(make_layout(3_ct, 3_ct),
                                       make_layout(make_tuple(2_c, 4_c), make_tuple(1_c, 8_c))))),
      "((3,(2,4)),(3,(2,2))):((177,(13,2)),(59,(26,1)))");
  EXPECT_EQ(
      toString(tiled_divide(make_layout(make_tuple(9_c, make_tuple(4_c, 8_c)),
                                        make_tuple(59_c, make_tuple(13_c, 1_c))),
                            make_tile(make_layout(3_ct, 3_ct),
                                      make_layout(make_tuple(2_c, 4_c), make_tuple(1_c, 8_c))))),
      "((3,(2,4)),3,(2,2)):((177,(13,2)),59,(26,1))");
  EXPECT_EQ(
      toString(flat_divide(make_layout(make_tuple(9_c, make_tuple(4_c, 8_c)),
                                       make_tuple(59_c, make_tuple(13_c, 1_c))),
                           make_tile(make_layout(3_ct, 3_ct),
                                     make_layout(make_tuple(2_c, 4_c), make_tuple(1_c, 8_c))))),
      "(3,(2,4),3,(2,2)):(177,(13,2),59,(26,1))");
  EXPECT_EQ(toString(zipped_divide(make_layout(make_tuple(64_d, 32_d), make_tuple(1_c, 64_d)),
                                   make_tile(make_layout(8_ct, 1_ct), make_layout(4_ct, 1_ct)))),
            "((8,4),(?8,?8)):((1,?64),(8,?256{div=4}))");
  EXPECT_EQ(toString(zipped_divide(
                make_layout(make_tuple(dyn(64, 8), dyn(32, 4)), make_tuple(1_c, dyn(64, 8))),
                make_tile(make_layout(8_ct, 1_ct), make_layout(4_ct, 1_ct)))),
            "((8,4),(?8,?8)):((1,?64{div=8}),(8,?256{div=32}))");
  EXPECT_EQ(toString(logical_divide(make_layout(dyn(64, 16), 1_ct), make_layout(dyn(16, 4), 1_ct))),
            "(?16{div=4},?4):(1,?16{div=4})");
  EXPECT_EQ(toString(logical_divide(make_layout(dyn(64, 16), 1_ct), make_layout(4_ct, 4_ct))),
            "(4,(4,?4)):(4,(1,16))");
  EXPECT_EQ(toString(tiled_divide(
                make_layout(make_tuple(dyn(64, 16), dyn(32, 8)), make_tuple(1_c, dyn(64, 16))),
                make_tile(make_layout(16_ct, 1_ct), make_layout(8_ct, 1_ct)))),
            "((16,8),?4,?4):((1,?64{div=16}),16,?512{div=128})");
}

TEST(LayoutParity, LogicalProduct) {
  EXPECT_EQ(toString(logical_product(make_layout(make_tuple(2_c, 2_c), make_tuple(4_c, 1_c)),
                                     make_layout(6_ct, 1_ct))),
            "((2,2),(2,3)):((4,1),(2,8))");
  EXPECT_EQ(toString(logical_product(make_layout(make_tuple(4_c, 2_c), make_tuple(1_c, 16_c)),
                                     make_layout(make_tuple(4_c, 2_c), make_tuple(2_c, 1_c)))),
            "((4,2),((2,2),2)):((1,16),((8,32),4))");
  EXPECT_EQ(toString(logical_product(make_layout(4_dt, 1_ct), make_layout(3_ct, 1_ct))),
            "(?4,3):(1,?4)");
  EXPECT_EQ(toString(logical_product(make_layout(make_tuple(2_c, 2_c), make_tuple(1_c, 2_c)),
                                     make_layout(3_dt, 1_ct))),
            "((2,2),?3):((1,2),4)");
  EXPECT_EQ(toString(logical_product(make_layout(1_ct, 0_ct), make_layout(1_ct, 0_ct))),
            "(1,1):(0,0)");
  EXPECT_EQ(toString(logical_product(make_layout(3_ct, 1_ct), make_layout(4_ct, 0_ct))),
            "(3,4):(1,0)");
  EXPECT_EQ(toString(logical_product(make_layout(3_ct, 0_ct), make_layout(4_ct, 1_ct))),
            "(3,4):(0,1)");
  EXPECT_EQ(toString(logical_product(make_layout(3_ct, 1_ct),
                                     make_layout(make_tuple(2_c, 4_c), make_tuple(1_c, 2_c)))),
            "(3,(2,4)):(1,(3,6))");
  EXPECT_EQ(toString(logical_product(make_layout(make_tuple(2_c, 4_c), make_tuple(1_c, 2_c)),
                                     make_layout(3_ct, 1_ct))),
            "((2,4),3):((1,2),8)");
  EXPECT_EQ(toString(logical_product(make_layout(make_tuple(8_c, make_tuple(2_c, 2_c)),
                                                 make_tuple(1_c, make_tuple(8_c, 16_c))),
                                     make_layout(4_ct, 2_ct))),
            "((8,(2,2)),4):((1,(8,16)),64)");
  EXPECT_EQ(toString(logical_product(make_layout(3_ct, 32_ct), make_layout(32_ct, 1_ct))),
            "(3,32):(32,1)");
  EXPECT_EQ(toString(logical_product(make_layout(3_ct, 2_ct), make_layout(4_ct, 1_ct))),
            "(3,(2,2)):(2,(1,6))");
  EXPECT_EQ(toString(logical_product(make_layout(3_ct, 32_ct), make_layout(128_ct, 1_ct))),
            "(3,(32,4)):(32,(1,96))");
  EXPECT_EQ(toString(logical_product(make_layout(3_ct, 32_ct),
                                     make_layout(make_tuple(8_c, 8_c), make_tuple(8_c, 1_c)))),
            "(3,((4,2),8)):(32,((8,96),1))");
  EXPECT_EQ(toString(logical_product(make_layout(make_tuple(4_c, 6_c), make_tuple(1_c, 6_c)),
                                     make_layout(3_ct, 1_ct))),
            "((4,6),3):((1,6),36)");
  // The tiler is laid out in the complement of the block, which skips the block's gaps.
  EXPECT_EQ(toString(logical_product(make_layout(make_tuple(4_c, 2_c), make_tuple(2_c, 14_c)),
                                     make_layout(2_ct, 2_ct))),
            "((4,2),2):((2,14),28)");
  EXPECT_EQ(toString(blocked_product(make_layout(make_tuple(2_c, 5_c), make_tuple(5_c, 1_c)),
                                     make_layout(make_tuple(3_c, 4_c), make_tuple(1_c, 3_c)))),
            "((2,3),(5,4)):((5,10),(1,30))");
  EXPECT_EQ(toString(raked_product(make_layout(make_tuple(2_c, 5_c), make_tuple(5_c, 1_c)),
                                   make_layout(make_tuple(3_c, 4_c), make_tuple(1_c, 3_c)))),
            "((3,2),(4,5)):((10,5),(30,1))");
  EXPECT_EQ(toString(zipped_product(make_layout(make_tuple(2_c, 5_c), make_tuple(5_c, 1_c)),
                                    make_layout(make_tuple(3_c, 4_c), make_tuple(1_c, 3_c)))),
            "((2,5),(3,4)):((5,1),(10,30))");
  EXPECT_EQ(toString(tiled_product(make_layout(make_tuple(2_c, 5_c), make_tuple(5_c, 1_c)),
                                   make_layout(make_tuple(3_c, 4_c), make_tuple(1_c, 3_c)))),
            "((2,5),3,4):((5,1),10,30)");
  EXPECT_EQ(toString(flat_product(make_layout(make_tuple(2_c, 5_c), make_tuple(5_c, 1_c)),
                                  make_layout(make_tuple(3_c, 4_c), make_tuple(1_c, 3_c)))),
            "(2,5,3,4):(5,1,10,30)");
  EXPECT_EQ(
      toString(zipped_product(make_layout(make_tuple(3_c, 4_c, 2_c), make_tuple(8_c, 2_c, 1_c)),
                              make_layout(make_tuple(2_c, 3_c, 5_c), make_tuple(1_c, 2_c, 6_c)))),
      "((3,4,2),(2,3,5)):((8,2,1),(24,48,144))");
  EXPECT_EQ(
      toString(tiled_product(make_layout(make_tuple(3_c, 4_c, 2_c), make_tuple(8_c, 2_c, 1_c)),
                             make_layout(make_tuple(2_c, 3_c, 5_c), make_tuple(1_c, 2_c, 6_c)))),
      "((3,4,2),2,3,5):((8,2,1),24,48,144)");
  EXPECT_EQ(
      toString(flat_product(make_layout(make_tuple(3_c, 4_c, 2_c), make_tuple(8_c, 2_c, 1_c)),
                            make_layout(make_tuple(2_c, 3_c, 5_c), make_tuple(1_c, 2_c, 6_c)))),
      "(3,4,2,2,3,5):(8,2,1,24,48,144)");
  EXPECT_EQ(toString(blocked_product(make_layout(make_tuple(3_c, 4_c), make_tuple(4_c, 1_c)),
                                     make_layout(make_tuple(2_c, 5_c), make_tuple(1_c, 2_c)))),
            "((3,2),(4,5)):((4,12),(1,24))");
  EXPECT_EQ(toString(raked_product(make_layout(make_tuple(3_c, 4_c), make_tuple(4_c, 1_c)),
                                   make_layout(make_tuple(2_c, 5_c), make_tuple(1_c, 2_c)))),
            "((2,3),(5,4)):((12,4),(24,1))");
  // Sources differ: CuTe C++ keeps the rank-1 tile group whole; cutegen unpacks it, as here.
  EXPECT_EQ(toString(tiled_product(make_layout(make_tuple(8_c), make_tuple(1_c)),
                                   make_layout(make_tuple(4_c), make_tuple(1_c)))),
            "((8),4):((1),8)");
  EXPECT_EQ(toString(flat_product(make_layout(make_tuple(8_c), make_tuple(1_c)),
                                  make_layout(make_tuple(4_c), make_tuple(1_c)))),
            "(8,4):(1,8)");
  // Sources differ: cutegen coalesces each mode of raked_product, as here; CuTe C++ keeps the
  // zipped modes, ((3,2),(1,2)):((4,1),(0,2)) here.
  EXPECT_EQ(toString(raked_product(make_layout(make_tuple(2_c, 2_c), make_tuple(1_c, 2_c)),
                                   make_layout(make_tuple(3_c), make_tuple(1_c)))),
            "((3,2),2):((4,1),2)");
  EXPECT_EQ(toString(raked_product(make_layout(make_tuple(4_c, 8_c), make_tuple(8_c, 1_c)),
                                   make_layout(make_tuple(1_c, 4_c), make_tuple(1_c, 1_c)))),
            "(4,(4,8)):(8,(32,1))");
  EXPECT_EQ(toString(raked_product(make_layout(make_tuple(2_c, 2_c), make_tuple(1_c, 2_c)),
                                   make_layout(make_tuple(2_c, make_tuple(2_c, 2_c)),
                                               make_tuple(1_c, make_tuple(2_c, 4_c))))),
            "((2,2),(4,2)):((4,1),(8,2))");
  // Sources differ: cutegen coalesces each mode, as here; CuTe C++ keeps the zipped modes,
  // ((4,128),(4,1)):((128,1),(512,0)) here.
  EXPECT_EQ(toString(raked_product(make_layout(128_ct, 1_ct),
                                   make_layout(make_tuple(4_c, 4_c), make_tuple(1_c, 4_c)))),
            "((4,128),4):((128,1),512)");
  EXPECT_EQ(toString(blocked_product(make_layout(make_tuple(2_c, 5_d), make_tuple(5_d, 1_c)),
                                     make_layout(make_tuple(3_c, 4_c), make_tuple(1_c, 3_c)))),
            "!unsupported-dynamic-complement:!unsupported-dynamic-complement");
  EXPECT_EQ(toString(tile_to_shape(make_layout(make_tuple(2_c, 2_c), make_tuple(1_c, 2_c)),
                                   make_tuple(6_c, 8_c))),
            "((2,3),(2,4)):((1,4),(2,12))");
  EXPECT_EQ(toString(tile_to_shape(make_layout(make_tuple(2_c, 2_c), make_tuple(1_c, 2_c)),
                                   make_tuple(6_c, 8_d))),
            "((2,3),(2,?4)):((1,4),(2,12))");
  EXPECT_EQ(toString(tile_to_shape(make_layout(make_tuple(4_c, 4_c), make_tuple(1_c, 4_c)),
                                   make_tuple(6_c, 8_c))),
            "!incompatible-shape-division:!incompatible-shape-division");
  EXPECT_EQ(toString(logical_product(make_layout(dyn(8, 4), 2_ct), make_layout(4_ct, 1_ct))),
            "(?8{div=4},(2,2)):(2,(1,?16{div=8}))");
  EXPECT_EQ(toString(logical_product(make_layout(make_tuple(2_c, 2_c), make_tuple(1_c, 2_c)),
                                     make_layout(dyn(8, 4), 2_ct))),
            "((2,2),?8{div=4}):((1,2),8)");
}

// upcast<N> divides the stride-1 mode and the other strides by N; downcast<N> multiplies them
// back.
TEST(LayoutParity, Recast) {
  EXPECT_EQ(toString(upcast(2, make_layout(make_tuple(4_c, 8_c), make_tuple(1_c, 4_c)))),
            "(2,8):(1,2)");
  EXPECT_EQ(toString(upcast(2, make_layout(make_tuple(4_c, 8_c), make_tuple(1_c, dyn(16, 8))))),
            "(2,8):(1,?8{div=4})");
  // A dynamic stride is divided with truncation.
  EXPECT_EQ(toString(upcast(4, make_layout(make_tuple(4_c, 8_c), make_tuple(1_c, 18_d)))),
            "(1,8):(1,?4)");
  EXPECT_EQ(toString(downcast(2, make_layout(make_tuple(4_c, 8_c), make_tuple(1_c, 4_c)))),
            "(8,8):(1,8)");
  EXPECT_EQ(toString(downcast(2, make_layout(make_tuple(4_c, 8_d), make_tuple(1_c, 4_d)))),
            "(8,?8):(1,?8{div=2})");
  // A negative stride keeps its sign: signum(s) * ceil_div(abs(s), N).
  EXPECT_EQ(toString(upcast(2, make_layout(8_ct, -4_ct))), "8:-2");
  EXPECT_EQ(
      toString(recast_layout(16, 32, make_layout(make_tuple(8_c, 2_c), make_tuple(1_c, 8_c)))),
      "(16,2):(1,16)");
  EXPECT_EQ(toString(recast_layout(
                64, 24, make_layout(make_tuple(32_c, 16_c, 8_c), make_tuple(1_c, 256_c, 32_c)))),
            "(12,16,8):(1,96,12)");
  EXPECT_EQ(toString(upcast(2, make_layout(8_ct, 3_ct))),
            "!invalid-recast-factor:!invalid-recast-factor");
  EXPECT_EQ(toString(upcast(
                2, make_layout(make_tuple(8_c, 16_c), make_tuple(E(1_ct, {0}), E(1_ct, {1}))))),
            "!unsupported-basis-operation:!unsupported-basis-operation");
  EXPECT_EQ(
      toString(upcast(4, make_layout(make_tuple(dyn(16, 8), 4_c), make_tuple(1_c, dyn(16, 8))))),
      "(?4{div=2},4):(1,?4{div=2})");
  EXPECT_EQ(toString(downcast(4, make_layout(make_tuple(4_c, 8_c), make_tuple(1_c, dyn(8, 2))))),
            "(16,8):(1,?32{div=8})");
  EXPECT_EQ(
      toString(downcast(2, make_layout(make_tuple(dyn(8, 4), 4_c), make_tuple(1_c, dyn(8, 4))))),
      "(?16{div=8},4):(1,?16{div=8})");
}

TEST(LayoutParity, CommonVectors) {
  EXPECT_EQ(toString(max_common_vector(make_layout(make_tuple(4_c, 8_c), make_tuple(1_c, 4_c)),
                                       make_layout(make_tuple(4_c, 8_c), make_tuple(1_c, 4_c)))),
            "32");
  EXPECT_EQ(toString(max_common_vector(make_layout(make_tuple(4_c, 8_c), make_tuple(1_c, 4_c)),
                                       make_layout(make_tuple(4_c, 8_c), make_tuple(1_c, 8_c)))),
            "4");
  EXPECT_EQ(toString(max_common_vector(make_layout(make_tuple(4_c, 8_d), make_tuple(1_c, 4_c)),
                                       make_layout(make_tuple(4_c, 8_d), make_tuple(1_c, 4_c)))),
            "4");
  EXPECT_EQ(toString(max_common_layout(make_layout(make_tuple(4_c, 8_c), make_tuple(1_c, 4_c)),
                                       make_layout(make_tuple(4_c, 8_c), make_tuple(1_c, 8_c)))),
            "4:1");
  EXPECT_EQ(toString(nullspace(make_layout(make_tuple(2_c, 4_c, 3_c), make_tuple(1_c, 0_c, 2_c)))),
            "4:2");
  EXPECT_EQ(toString(max_alignment(make_layout(make_tuple(2_c, 4_c), make_tuple(4_c, 1_c)))), "8");
  EXPECT_EQ(toString(max_alignment(make_layout(4_ct, 2_ct))), "1");
  EXPECT_EQ(toString(max_alignment(make_layout(make_tuple(2_c, 3_c), make_tuple(1_c, 2_c)))), "6");
  EXPECT_EQ(toString(max_alignment(make_layout(make_tuple(8_c, 4_c), make_tuple(1_c, 16_c)))), "8");
  // A dynamic mode ends the contiguous prefix.
  EXPECT_EQ(toString(max_alignment(make_layout(make_tuple(8_c, 4_d), make_tuple(1_c, 32_d)))), "8");
  EXPECT_EQ(toString(nullspace(make_layout(make_tuple(2_c, 3_c), make_tuple(1_c, 0_c)))), "3:2");
  EXPECT_EQ(toString(nullspace(make_layout(1_ct, 0_ct))), "1:0");
  EXPECT_EQ(toString(nullspace(make_layout(make_tuple(1_c, 4_c, 6_c), make_tuple(0_c, 1_c, 0_c)))),
            "(1,6):(0,4)");
  // Only a static zero stride is in the nullspace.
  EXPECT_EQ(toString(nullspace(make_layout(8_ct, 0_dt))), "1:0");
}

// Cases where Fly does not reproduce the reference; the expected results are the reference's.  Run
// with --gtest_also_run_disabled_tests to see them.
TEST(LayoutParity, DISABLED_KnownDivergences) {
  // CuTe C++ returns a runtime first coordinate; Fly folds a multiple of 8 modulo 4 to a static 0.
  EXPECT_EQ(
      toString(make_layout(make_tuple(4_c, 8_c), make_tuple(1_c, 4_c)).get_hier_coord(dyn(24, 8))),
      "(?0,?6{div=2})");
}
