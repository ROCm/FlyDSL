//===- IntTupleTest.cpp - IntTuple reference results -----------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// IntTuple and the tuple algorithms of IntTupleAlgorithms.hpp.
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

using namespace core_test;

// rank and depth only see the tree, so they are static even for dynamic leaves; a rank-1 tuple
// is one level deeper than its element.
TEST(IntTupleParity, RankAndDepth) {
  EXPECT_EQ(rank(8_dt), 1);
  EXPECT_EQ(rank(make_tuple(8_d)), 1);
  EXPECT_EQ(rank(make_tuple(make_tuple(2_c, 3_d), make_tuple(4_c, make_tuple(5_c, 6_d)))), 2);
  EXPECT_EQ(depth(8_ct), 0);
  EXPECT_EQ(depth(make_tuple(8_c)), 1);
  EXPECT_EQ(depth(make_tuple(make_tuple(8_c))), 2);
  EXPECT_EQ(depth(make_tuple(make_tuple(2_c, 3_d), make_tuple(4_c, make_tuple(5_c, 6_d)))), 3);
}

// A reduction is static only when every leaf it reads is, except that a static zero factor absorbs
// dynamic ones.  Divisibilities multiply under product and meet (gcd) under sum.
TEST(IntTupleParity, ProductSumAndSize) {
  EXPECT_EQ(toString(product(make_tuple(2_c, make_tuple(3_c, 4_c)))), "24");
  EXPECT_EQ(toString(product(make_tuple(2_c, make_tuple(3_d, 4_c)))), "?24{div=8}");
  EXPECT_EQ(toString(product(make_tuple(2_d, 3_d))), "?6");
  EXPECT_EQ(toString(product(make_tuple(dyn(8, 4), make_tuple(dyn(6, 2), 3_c)))), "?144{div=24}");
  EXPECT_EQ(toString(product(make_tuple(7_d, 0_c))), "0");
  EXPECT_EQ(toString(product(make_tuple(1_c, 5_d))), "?5");
  EXPECT_EQ(toString(product(make_tuple<Runtime>())), "1");
  EXPECT_EQ(toString(size(make_tuple(make_tuple(2_c, 3_d), 4_c))), "?24{div=8}");
  EXPECT_EQ(toString(sum(make_tuple(2_c, make_tuple(3_d, 4_c)))), "?9");
  EXPECT_EQ(toString(sum(make_tuple(dyn(8, 4), dyn(12, 6)))), "?20{div=2}");
  EXPECT_EQ(toString(product_each(make_tuple(2_c, make_tuple(3_d, 4_c), make_tuple(5_c, 6_c)))),
            "(2,?12{div=4},30)");
  EXPECT_EQ(toString(product_each(8_dt)), "(?8)");
}

TEST(IntTupleParity, InnerProduct) {
  EXPECT_EQ(toString(inner_product(make_tuple(2_c, make_tuple(3_c, 4_c)),
                                   make_tuple(1_c, make_tuple(5_c, 6_c)))),
            "41");
  EXPECT_EQ(toString(inner_product(make_tuple(2_d, make_tuple(3_c, 4_c)),
                                   make_tuple(1_c, make_tuple(5_c, 6_d)))),
            "?41");
  // A static zero stride removes the dynamic coordinate.
  EXPECT_EQ(toString(inner_product(make_tuple(2_d, 3_c), make_tuple(0_c, 1_c))), "3");
  EXPECT_EQ(toString(inner_product(make_tuple(dyn(8, 4), 3_c), make_tuple(2_c, dyn(6, 2)))),
            "?34{div=2}");
}

TEST(IntTupleParity, Flatten) {
  EXPECT_EQ(
      toString(flatten(make_tuple(make_tuple(2_c, 3_d), make_tuple(4_c, make_tuple(5_c, 6_d))))),
      "(2,?3,4,5,?6)");
  EXPECT_EQ(toString(flatten(7_dt)), "?7");
  EXPECT_EQ(toString(flatten(make_tuple(make_tuple(7_c)))), "(7)");
  EXPECT_EQ(toString(unflatten(make_tuple(2_c, 3_d, 4_c, 5_c),
                               make_tuple(make_tuple(1_c, 1_c), make_tuple(1_c, 1_c)))),
            "((2,?3),(4,5))");
}

TEST(IntTupleParity, WrapUnwrapReverseZip) {
  EXPECT_EQ(toString(wrap(7_dt)), "(?7)");
  EXPECT_EQ(toString(wrap(make_tuple(2_c, 3_c))), "(2,3)");
  EXPECT_EQ(toString(unwrap(make_tuple(make_tuple(make_tuple(7_d))))), "?7");
  EXPECT_EQ(toString(unwrap(make_tuple(make_tuple(2_c, 3_c)))), "(2,3)");
  EXPECT_EQ(toString(reverse(make_tuple(2_c, make_tuple(3_c, 4_d), 5_c))), "(5,(3,?4),2)");
  EXPECT_EQ(toString(zip(make_tuple(make_tuple(2_c, 3_c), make_tuple(4_d, 5_c)))),
            "((2,?4),(3,5))");
}

TEST(IntTupleParity, TakeSelectGroup) {
  EXPECT_EQ(toString(take<1, 3>(make_tuple(2_c, 3_d, make_tuple(4_c, 5_c), 6_c))), "(?3,(4,5))");
  // take of one mode is still a 1-tuple.
  EXPECT_EQ(toString(take<1, 2>(make_tuple(2_c, 3_d, 4_c))), "(?3)");
  EXPECT_EQ(toString(select<3, 0>(make_tuple(2_c, 3_d, make_tuple(4_c, 5_c), 6_c))), "(6,2)");
  EXPECT_EQ(toString(group<1, 3>(make_tuple(2_c, 3_d, 4_c, 5_c))), "(2,(?3,4),5)");
  EXPECT_EQ(toString(group<1, 2>(make_tuple(2_c, 3_d, 4_c))), "(2,(?3),4)");
}

TEST(IntTupleParity, AppendPrependReplace) {
  EXPECT_EQ(toString(append(make_tuple(2_c, 3_d), make_tuple(4_c, 5_c))), "(2,?3,(4,5))");
  EXPECT_EQ(toString(append<4>(make_tuple(2_c, 3_d), 1_ct)), "(2,?3,1,1)");
  EXPECT_EQ(toString(prepend(make_tuple(2_c, 3_d), 4_dt)), "(?4,2,?3)");
  EXPECT_EQ(toString(replace<1>(make_tuple(2_c, 3_d, 4_c), make_tuple(5_c, 6_c))), "(2,(5,6),4)");
  EXPECT_EQ(toString(insert<1>(make_tuple(2_c, 3_d, 4_c), 9_dt)), "(2,?9,?3,4)");
  EXPECT_EQ(toString(remove<1>(make_tuple(2_c, 3_d, 4_c))), "(2,4)");
  EXPECT_EQ(toString(repeat<3>(make_tuple(2_c, 3_d))), "((2,?3),(2,?3),(2,?3))");
  EXPECT_EQ(toString(repeat(8_ct, 1)), "8");
  EXPECT_EQ(toString(tuple_repeat(8_ct, 1)), "(8)");
  EXPECT_EQ(toString(tuple_repeat(make_tuple(2_c, 3_d), 1)), "((2,?3))");
  EXPECT_EQ(toString(tuple_repeat(make_tuple(2_c, 3_d), 2)), "((2,?3),(2,?3))");
  EXPECT_EQ(
      toString(repeat_like(
          make_tuple(make_tuple(1_c, 2_c), 3_c, make_tuple(4_c, make_tuple(5_c, 6_c))), 10_dt)),
      "((?10,?10),?10,(?10,(?10,?10)))");
}

// congruent and weakly_congruent are structural; compatible compares sizes and is a runtime answer
// when a size is dynamic.
TEST(IntTupleParity, Profiles) {
  EXPECT_TRUE(
      congruent(make_tuple(2_c, make_tuple(3_c, 4_d)), make_tuple(5_d, make_tuple(6_c, 7_c))));
  EXPECT_FALSE(congruent(make_tuple(2_c, make_tuple(3_c, 4_c)), make_tuple(5_c, 6_c)));
  EXPECT_TRUE(weakly_congruent(make_tuple(2_c, 3_c),
                               make_tuple(make_tuple(4_c, 5_c), make_tuple(6_c, 7_c))));
  EXPECT_FALSE(weakly_congruent(make_tuple(make_tuple(4_c, 5_c), 3_c), make_tuple(2_c, 3_c)));
  EXPECT_EQ(toString(compatible(make_tuple(12_c, 6_c),
                                make_tuple(make_tuple(3_c, 4_c), make_tuple(2_c, 3_c)))),
            "1");
  EXPECT_EQ(toString(compatible(make_tuple(12_c, 6_c),
                                make_tuple(make_tuple(3_c, 4_c), make_tuple(2_c, 2_c)))),
            "0");
  EXPECT_EQ(toString(compatible(make_tuple(12_d, 6_c),
                                make_tuple(make_tuple(3_c, 4_c), make_tuple(2_c, 3_c)))),
            "?1");
}

TEST(IntTupleParity, Comparisons) {
  EXPECT_EQ(toString(lex_less(make_tuple(1_c, 2_c), make_tuple(1_c, 3_c))), "1");
  EXPECT_EQ(toString(lex_less(make_tuple(1_d, 2_c), make_tuple(1_c, 3_c))), "?1");
  EXPECT_EQ(toString(colex_less(make_tuple(2_c, 1_c), make_tuple(1_c, 3_c))), "1");
  EXPECT_EQ(toString(elem_less(make_tuple(1_c, 2_c), make_tuple(2_c, 3_c))), "1");
  EXPECT_EQ(toString(elem_less(make_tuple(1_d, 2_c), make_tuple(2_c, 3_c))), "?1");
  EXPECT_EQ(toString(elem_less(make_tuple(dyn(8, 4), 2_c), make_tuple(8_c, 3_c))), "?0");
  // A mode of a beyond the rank of b makes a not less, whatever its value.
  EXPECT_EQ(toString(elem_less(make_tuple(1_c, 5_d), make_tuple(2_c))), "0");
}

// A shorter divisor is padded with ones; a leaf divisor is consumed across the modes.
TEST(IntTupleParity, CeilDiv) {
  EXPECT_EQ(toString(ceil_div(make_tuple(7_c, make_tuple(9_c, 5_c)),
                              make_tuple(4_c, make_tuple(8_c, 2_c)))),
            "(2,(2,3))");
  EXPECT_EQ(toString(ceil_div(make_tuple(6_c, make_tuple(4_c, 5_c)), 16_ct)), "(1,(2,5))");
  EXPECT_EQ(toString(ceil_div(make_tuple(7_c, 9_c, 5_c), make_tuple(4_c, 8_c))), "(2,2,5)");
  EXPECT_EQ(toString(ceil_div(make_tuple(7_d, 9_c), make_tuple(4_c, 8_d))), "(?2,?2)");
  EXPECT_EQ(toString(ceil_div(make_tuple(dyn(16, 8), 9_c), make_tuple(4_c, 3_c))), "(?4{div=2},3)");
  EXPECT_EQ(toString(round_up(make_tuple(7_c, 9_c, 5_c), make_tuple(4_c, 8_c))), "(8,16,5)");
  EXPECT_EQ(toString(round_up(make_tuple(7_d, 9_c), make_tuple(4_c, 8_d))), "(?8{div=4},?16)");
  EXPECT_EQ(toString(round_up(make_tuple(dyn(12, 4), 9_c), make_tuple(8_c, 4_c))),
            "(?16{div=8},12)");
  EXPECT_EQ(toString(ceil_div(make_tuple(4_c, dyn(16, 8)), 8_ct)), "(1,?8{div=4})");
  // Once a dynamic mode consumes part of a leaf divisor, the rest of it is dynamic.
  EXPECT_EQ(toString(ceil_div(make_tuple(dyn(16, 8), 9_c), 32_ct)), "(?1,?5)");
}

// shape_div requires one operand to divide the other.
TEST(IntTupleParity, ShapeDiv) {
  EXPECT_EQ(toString(shape_div(make_tuple(3_c, 4_c), 6_ct)), "(1,2)");
  EXPECT_EQ(toString(shape_div(make_tuple(make_tuple(3_c, 4_c), 6_c), 36_ct)), "((1,1),2)");
  EXPECT_EQ(toString(shape_div(make_tuple(6_c, make_tuple(3_c, 4_c)), 36_ct)), "(1,(1,2))");
  EXPECT_EQ(toString(shape_div(make_tuple(3_c, 4_c), 12_ct)), "(1,1)");
  EXPECT_EQ(toString(shape_div(make_tuple(64_c, 64_c), make_tuple(8_c, 32_c))), "(8,2)");
  EXPECT_EQ(toString(shape_div(make_tuple(12_d, 8_d), make_tuple(4_c, 2_c))), "(?3,?4)");
  EXPECT_EQ(toString(shape_div(make_tuple(dyn(16, 8), 6_c), make_tuple(4_c, 3_c))),
            "(?4{div=2},2)");
  EXPECT_EQ(toString(shape_div(make_tuple(6_c, 4_c), make_tuple(4_c, 3_c))),
            "!incompatible-shape-division");
  EXPECT_EQ(toString(shape_div(make_tuple(dyn(16, 8), 6_c), 32_ct)), "(?1,?3)");
  EXPECT_EQ(toString(shape_div(make_tuple(dyn(16, 8), 6_c), 4_ct)), "(?4{div=2},?6)");
}

// Tuple + and - treat a missing mode as zero.
TEST(IntTupleParity, ElementwiseArithmetic) {
  EXPECT_EQ(toString(make_tuple(1_c, make_tuple(2_c)) + make_tuple(3_c, make_tuple(4_c, 5_c), 6_c)),
            "(4,(6,5),6)");
  EXPECT_EQ(toString(make_tuple(1_d, 0_c, 2_c) + make_tuple(0_c, make_tuple(3_c, 4_d), 0_c)),
            "(?1,(3,?4),2)");
  EXPECT_EQ(toString(E(1_ct, {0}) + E(2_ct, {1})), "(1,2)");
  EXPECT_EQ(toString(make_tuple(1_c, make_tuple(2_c)) - make_tuple(3_c, make_tuple(4_c, 5_c), 6_c)),
            "(-2,(-2,-5),-6)");
  EXPECT_EQ(
      toString(make_tuple(8_c, make_tuple(15_c, 24_d)) * make_tuple(2_c, make_tuple(3_c, 4_c))),
      "(16,(45,?96{div=4}))");
  EXPECT_EQ(
      toString(make_tuple(8_c, make_tuple(15_c, 24_d)) / make_tuple(2_c, make_tuple(3_c, 4_c))),
      "(4,(5,?6))");
  EXPECT_EQ(
      toString(make_tuple(8_c, make_tuple(15_c, 24_d)) % make_tuple(3_c, make_tuple(4_c, 5_c))),
      "(2,(3,?4))");
  EXPECT_EQ(toString(elem_scale(make_tuple(2_c, make_tuple(3_c, 4_d)),
                                make_tuple(5_c, make_tuple(6_c, 7_c)))),
            "(10,(18,?28{div=7}))");
  EXPECT_EQ(toString(make_tuple(dyn(8, 4), 3_c) + make_tuple(dyn(12, 6), 5_d)), "(?20{div=2},?8)");
  EXPECT_EQ(toString(make_tuple(dyn(16, 8), 24_d) / make_tuple(4_c, 4_c)), "(?4{div=2},?6)");
  EXPECT_EQ(toString(elem_scale(make_tuple(dyn(8, 4), 3_c), make_tuple(3_c, dyn(6, 2)))),
            "(?24{div=12},?18{div=6})");
}

// A static-1 mode gets stride 0; a dynamic shape makes every later stride dynamic and passes its
// divisibility on.
TEST(IntTupleParity, CompactColMajor) {
  EXPECT_EQ(toString(compact_col_major(make_tuple(4_c, 8_c, 2_c))), "(1,4,32)");
  EXPECT_EQ(toString(compact_col_major(make_tuple(4_c, 8_c, 1_c, 2_c))), "(1,4,0,32)");
  EXPECT_EQ(
      toString(compact_col_major(make_tuple(4_c, make_tuple(8_c, 1_c, make_tuple(2_c, 3_c))))),
      "(1,(4,0,(32,64)))");
  EXPECT_EQ(toString(compact_col_major(make_tuple(4_c, 8_d, 2_c))), "(1,4,?32{div=4})");
  // Only a static 1 is dropped.
  EXPECT_EQ(toString(compact_col_major(make_tuple(1_d, 4_c))), "(1,?1)");
  EXPECT_EQ(toString(compact_col_major(make_tuple(dyn(8, 2), 4_c, 16_c, 2_c))),
            "(1,?8{div=2},?32{div=8},?512{div=128})");
  EXPECT_EQ(toString(compact_col_major(8_dt)), "1");
  EXPECT_EQ(toString(compact_col_major(1_ct)), "0");
  EXPECT_EQ(toString(compact_col_major(make_tuple(make_tuple(4_c, 8_c), 1_c, 2_c))),
            "((1,4),0,32)");
  EXPECT_EQ(toString(compact_col_major(make_tuple(4_c, 8_d, 2_c), 3_ct)), "(3,12,?96{div=12})");
  EXPECT_EQ(toString(compact_col_major(make_tuple(4_c, 8_c), 3_dt)), "(?3,?12{div=4})");
}

TEST(IntTupleParity, CompactRowMajor) {
  EXPECT_EQ(toString(compact_row_major(make_tuple(4_c, 8_c, 2_c))), "(16,2,1)");
  EXPECT_EQ(
      toString(compact_row_major(make_tuple(4_c, make_tuple(8_c, 1_c, make_tuple(2_c, 3_c))))),
      "(48,(6,0,(3,1)))");
  EXPECT_EQ(toString(compact_row_major(make_tuple(4_c, 8_c, 1_c, 2_c))), "(16,2,0,1)");
  EXPECT_EQ(toString(compact_row_major(make_tuple(make_tuple(4_c, 8_c), 1_c, 2_c))),
            "((16,2),0,1)");
  // The outermost dynamic mode reaches no stride.
  EXPECT_EQ(toString(compact_row_major(make_tuple(4_d, 8_c, 2_c))), "(16,2,1)");
  EXPECT_EQ(toString(compact_row_major(make_tuple(4_c, dyn(8, 4), 2_c))), "(?16{div=8},2,1)");
}

// The order gives each mode its rank, innermost first.
TEST(IntTupleParity, CompactOrder) {
  EXPECT_EQ(toString(compact_order(make_tuple(2_c, 3_c, 4_c), make_tuple(2_c, 0_c, 1_c))),
            "(12,1,3)");
  EXPECT_EQ(toString(compact_order(make_tuple(2_c, make_tuple(3_c, 4_c)),
                                   make_tuple(2_c, make_tuple(1_c, 0_c)))),
            "(12,(4,1))");
  EXPECT_EQ(toString(compact_order(make_tuple(2_d, 3_c, 4_d), make_tuple(1_c, 2_c, 0_c))),
            "(?4,?8,1)");
  EXPECT_EQ(toString(compact_order(make_tuple(dyn(8, 2), 4_c, 16_c, 2_c),
                                   make_tuple(1_c, 2_c, 3_c, 0_c))),
            "(2,?16{div=4},?64{div=16},1)");
  EXPECT_EQ(toString(compact_order(make_tuple(8_c, dyn(4, 2), 16_c, 2_c),
                                   make_tuple(1_c, 2_c, 3_c, 0_c))),
            "(2,16,?64{div=32},1)");
  EXPECT_EQ(toString(compact_order(make_tuple(8_c, dyn(4, 2), 16_c, dyn(2, 2)),
                                   make_tuple(1_c, 2_c, 3_c, 0_c))),
            "(?2{div=2},?16{div=16},?64{div=32},1)");
  EXPECT_EQ(toString(compact_order(make_tuple(1_c, 4_c, 1_d, 32_c, 1_c),
                                   make_tuple(3_c, 0_c, 2_c, 4_c, 1_c))),
            "(0,1,4,?4{div=4},0)");
  EXPECT_EQ(toString(compact_order(make_tuple(1_c, 4_c, 1_d, 32_c, 1_c),
                                   make_tuple(1_c, 0_c, 4_c, 3_c, 2_c))),
            "(0,1,128,4,0)");
}

// A coordinate is static only if the index and every shape it divides by are.
TEST(IntTupleParity, Idx2Crd) {
  EXPECT_EQ(toString(idx2crd(16_ct, make_tuple(make_tuple(2_c, 3_c), 4_c))), "((0,2),2)");
  EXPECT_EQ(toString(idx2crd(16_dt, make_tuple(make_tuple(2_c, 3_c), 4_c))), "((?0,?2),?2)");
  EXPECT_EQ(toString(idx2crd(16_ct, make_tuple(make_tuple(2_c, 3_d), 4_c))), "((?0,?2),?2)");
  EXPECT_EQ(toString(idx2crd(make_tuple(5_c, 3_c), make_tuple(make_tuple(2_c, 3_c), 4_c))),
            "((1,2),3)");
  EXPECT_EQ(toString(idx2crd(13_dt, make_tuple(2_c, 3_c, 4_c), make_tuple(1_c, 2_c, 6_c))),
            "(?1,?0,?2)");
  // An index past the shape carries into the last mode.
  EXPECT_EQ(toString(idx2crd(47_ct, make_tuple(2_c, 3_c, 4_c))), "(1,2,7)");
  EXPECT_EQ(toString(idx2crd(13_ct, make_tuple(make_tuple(2_c, 2_c), 3_c))), "((1,0),3)");
  EXPECT_EQ(toString(idx2crd(dyn(24, 8), make_tuple(16_c, 4_c))), "(?8{div=8},?1)");
  EXPECT_EQ(toString(idx2crd(dyn(40, 8), make_tuple(16_c, dyn(4, 2)))), "(?8{div=8},?2)");
}

TEST(IntTupleParity, Crd2Idx) {
  EXPECT_EQ(toString(crd2idx(make_tuple(make_tuple(1_c, 2_c), 3_c),
                             make_tuple(make_tuple(2_c, 3_c), 4_c))),
            "23");
  EXPECT_EQ(toString(crd2idx(make_tuple(make_tuple(1_d, 2_c), 3_c),
                             make_tuple(make_tuple(2_c, 3_d), 4_c))),
            "?23");
  // A scalar index passes through.
  EXPECT_EQ(toString(crd2idx(17_dt, make_tuple(make_tuple(2_c, 3_c), 4_c))), "?17");
  EXPECT_EQ(
      toString(crd2idx(make_tuple(1_c, make_tuple(2_d, 3_c)), make_tuple(2_c, make_tuple(3_c, 4_c)),
                       make_tuple(12_c, make_tuple(1_c, 3_c)))),
      "?23");
  EXPECT_EQ(toString(crd2idx(make_tuple(1_d, 2_d), make_tuple(4_c, 8_c), make_tuple(0_c, 1_c))),
            "?2");
  EXPECT_EQ(toString(crd2idx(make_tuple(1_d, 2_d), make_tuple(4_c, 8_c),
                             make_tuple(dyn(8, 8), dyn(16, 4)))),
            "?40{div=4}");
  EXPECT_EQ(toString(crd2crd(make_tuple(1_c, 2_c), make_tuple(2_c, 3_c), make_tuple(3_c, 2_c))),
            "(1,2)");
  EXPECT_EQ(toString(crd2idx(make_tuple(dyn(4, 4), 1_d), make_tuple(8_c, 4_c))), "?12{div=4}");
  EXPECT_EQ(toString(crd2idx(make_tuple(1_c, 2_c), make_tuple(dyn(8, 4), 4_c))), "?17");
  EXPECT_EQ(toString(crd2idx(make_tuple(dyn(4, 2), dyn(6, 3)), make_tuple(8_c, 8_c),
                             make_tuple(dyn(8, 4), 6_c))),
            "?68{div=2}");
}

TEST(IntTupleParity, SliceDiceFilterZeros) {
  EXPECT_EQ(toString(slice(make_tuple(_, make_tuple(1_c, _), 2_c),
                           make_tuple(2_c, make_tuple(3_c, 4_d), 5_c))),
            "(2,?4)");
  EXPECT_EQ(toString(slice(_, make_tuple(2_c, 3_c))), "(2,3)");
  EXPECT_EQ(toString(dice(make_tuple(_, make_tuple(1_c, _), 2_c),
                          make_tuple(2_c, make_tuple(3_c, 4_d), 5_c))),
            "(3,5)");
  EXPECT_EQ(toString(filter_zeros(make_tuple(0_c, make_tuple(1_c, 0_c), 2_d),
                                  make_tuple(2_c, make_tuple(3_c, 4_d), 5_c))),
            "(1,(3,1),5)");
  EXPECT_EQ(toString(product_like(make_tuple(make_tuple(2_c, 3_c), make_tuple(4_c, 5_c)),
                                  make_tuple(1_c, make_tuple(1_c, 1_c)))),
            "(6,(4,5))");
}

TEST(IntTupleParity, Basis) {
  EXPECT_EQ(toString(make_basis_like(make_tuple(2_c, make_tuple(3_c, 4_c)))),
            "(1E0,(1E1E0,1E1E1))");
}

// Cases where Fly does not reproduce the reference; the expected results are the reference's.  Run
// with --gtest_also_run_disabled_tests to see them.
TEST(IntTupleParity, DISABLED_KnownDivergences) {
  // CuTe C++ returns a runtime first coordinate; Fly folds a multiple of 8 modulo 4 to a static 0.
  EXPECT_EQ(toString(idx2crd(dyn(24, 8), make_tuple(4_c, 8_c))), "(?0,?6{div=2})");
}
