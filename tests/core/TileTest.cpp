//===- TileTest.cpp - Tile reference results -------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

using namespace core_test;

// The size of a tiler is the product of its modes.
TEST(TileParity, Size) {
  EXPECT_EQ(toString(size(make_tile(make_layout(2_ct, 3_ct),
                                    make_layout(make_tuple(2_c, 4_c), make_tuple(1_c, 8_c))))),
            "16");
  EXPECT_EQ(toString(size(make_tile(make_tile(make_layout(2_ct, 1_ct), make_layout(3_ct, 1_ct)),
                                    make_layout(4_ct, 2_ct)))),
            "24");
  EXPECT_EQ(toString(size(make_tile(make_layout(4_dt, 1_ct), make_layout(8_ct, 1_ct)))),
            "?32{div=8}");
  EXPECT_EQ(toString(size(make_tile(make_layout(dyn(16, 4), 1_ct), make_layout(8_ct, 1_ct)))),
            "?128{div=32}");
}

// Each tiler mode composes with the matching layout mode and an underscore keeps its mode whole;
// composition keeps only the tiled modes.
TEST(TileParity, ByModeComposition) {
  EXPECT_EQ(toString(composition(make_layout(make_tuple(12_c, make_tuple(4_c, 8_c)),
                                             make_tuple(59_c, make_tuple(13_c, 1_c))),
                                 make_tile(make_layout(3_ct, 4_ct), make_layout(8_ct, 2_ct)))),
            "(3,(2,4)):(236,(26,1))");
  EXPECT_EQ(
      toString(composition(make_layout(make_tuple(12_c, 4_c, 8_c), make_tuple(59_c, 13_c, 1_c)),
                           make_tile(make_layout(3_ct, 1_ct)))),
      "(3):(59)");
  EXPECT_EQ(toString(composition(make_layout(make_tuple(12_c, make_tuple(4_c, 8_c)),
                                             make_tuple(59_c, make_tuple(13_c, 1_c))),
                                 make_tile(_, make_layout(8_ct, 2_ct)))),
            "(12,(2,4)):(59,(26,1))");
  EXPECT_EQ(
      toString(composition(make_layout(make_tuple(make_tuple(4_c, 6_c), 8_c),
                                       make_tuple(make_tuple(1_c, 4_c), 24_c)),
                           make_tile(make_tile(make_layout(2_ct, 1_ct), make_layout(3_ct, 2_ct)),
                                     make_layout(4_ct, 2_ct)))),
      "((2,3),4):((1,8),48)");
  EXPECT_EQ(toString(composition(make_layout(make_tuple(12_c, make_tuple(4_c, 8_c)),
                                             make_tuple(59_c, make_tuple(13_c, 1_c))),
                                 make_tuple(3_c, 8_c))),
            "(3,(4,2)):(59,(13,1))");
  EXPECT_EQ(toString(composition(make_layout(make_tuple(12_d, make_tuple(4_d, 8_d)),
                                             make_tuple(59_d, make_tuple(13_d, 1_c))),
                                 make_tile(make_layout(3_ct, 4_ct), make_layout(8_ct, 2_ct)))),
            "(3,(?2,?4)):(?236{div=4},(?26{div=2},?1))");
  EXPECT_EQ(toString(composition(make_layout(make_tuple(12_c, make_tuple(4_c, 8_c)),
                                             make_tuple(59_c, make_tuple(13_c, 1_c))),
                                 make_tile(make_layout(3_dt, 1_ct), make_layout(8_ct, 1_ct)))),
            "(?3,(4,2)):(59,(13,1))");
  EXPECT_EQ(
      toString(composition(make_layout(make_tuple(dyn(64, 16), 32_c), make_tuple(1_c, dyn(64, 16))),
                           make_tile(make_layout(8_ct, 2_ct), make_layout(4_ct, 8_ct)))),
      "(8,4):(2,?512{div=128})");
  EXPECT_EQ(toString(composition(make_layout(make_tuple(64_c, 32_c), make_tuple(1_c, 64_c)),
                                 make_tile(make_layout(dyn(16, 4), 2_ct), _))),
            "(?16{div=4},32):(2,64)");
  EXPECT_EQ(toString(composition(make_layout(make_tuple(64_c, 32_c), make_tuple(1_c, 64_c)),
                                 make_tile(make_layout(8_ct, dyn(4, 4)), _))),
            "(8,32):(?4{div=4},64)");
}

// A tiler with fewer modes than the layout leaves the trailing modes untouched.
TEST(TileParity, Divide) {
  EXPECT_EQ(
      toString(logical_divide(make_layout(make_tuple(8_c, 6_c, 4_c), make_tuple(1_c, 8_c, 48_c)),
                              make_tile(make_layout(2_ct, 1_ct)))),
      "((2,4),6,4):((1,2),8,48)");
  EXPECT_EQ(
      toString(logical_divide(make_layout(make_tuple(8_c, 6_c, 4_c), make_tuple(1_c, 8_c, 48_c)),
                              make_tile(_, make_layout(3_ct, 2_ct)))),
      "(8,(3,2),4):(1,(16,8),48)");
  EXPECT_EQ(toString(logical_divide(make_layout(make_tuple(8_c, 6_c), make_tuple(1_c, 8_c)),
                                    make_tuple(4_c, 3_c))),
            "((4,2),(3,2)):((1,4),(8,24))");
  EXPECT_EQ(toString(zipped_divide(make_layout(make_tuple(8_c, 32_c), make_tuple(1_c, 8_c)),
                                   make_tuple(4_c, 2_c))),
            "((4,2),(2,16)):((1,8),(4,16))");
  EXPECT_EQ(
      toString(zipped_divide(make_layout(make_tuple(make_tuple(4_c, 4_c), 32_c),
                                         make_tuple(make_tuple(1_c, 4_c), 16_c)),
                             make_tile(make_tile(make_layout(2_ct, 1_ct), make_layout(2_ct, 1_ct)),
                                       make_layout(4_ct, 1_ct)))),
      "(((2,2),4),((2,2),8)):(((1,4),16),((2,8),64))");
  EXPECT_EQ(
      toString(tiled_divide(make_layout(make_tuple(8_c, 16_d, 4_c), make_tuple(1_c, 8_c, 128_d)),
                            make_tile(make_layout(2_ct, 1_ct), make_layout(4_ct, 1_ct)))),
      "((2,4),4,?4,4):((1,8),2,32,?128)");
  EXPECT_EQ(toString(logical_divide(make_layout(make_tuple(64_c, 32_c), make_tuple(1_c, 64_c)),
                                    make_tile(make_layout(dyn(16, 4), 1_ct), _))),
            "((?16{div=4},?4),32):((1,?16{div=4}),64)");
  EXPECT_EQ(toString(logical_divide(make_layout(make_tuple(8_c, 6_c), make_tuple(dyn(4, 4), 1_c)),
                                    make_tile(make_layout(2_ct, 1_ct), make_layout(3_ct, 1_ct)))),
            "((2,4),(3,2)):((?4{div=4},?8{div=8}),(1,3))");
  EXPECT_EQ(toString(zipped_divide(
                make_layout(make_tuple(dyn(128, 16), dyn(64, 8)), make_tuple(1_c, dyn(128, 16))),
                make_tile(make_layout(16_ct, 1_ct), make_layout(8_ct, 1_ct)))),
            "((16,8),(?8,?8)):((1,?128{div=16}),(16,?1024{div=128}))");
  EXPECT_EQ(toString(flat_divide(make_layout(make_tuple(dyn(32, 8), 16_c), make_tuple(16_c, 1_c)),
                                 make_tile(make_layout(8_ct, 1_ct), make_layout(4_ct, 1_ct)))),
            "(8,4,?4,4):(16,1,128,4)");
  // A multiple of 8 divided by 4 is still a multiple of 2.
  EXPECT_EQ(
      toString(logical_divide(make_layout(make_tuple(dyn(64, 8), 6_c), make_tuple(1_c, dyn(64, 8))),
                              make_tile(make_layout(4_ct, 1_ct), make_layout(3_ct, 1_ct)))),
      "((4,?16{div=2}),(3,2)):((1,4),(?64{div=8},?192{div=24}))");
  // tile_unzip regroups a divided layout by the tiler: the first part of each tiled mode joins the
  // tile group, the other parts and the untiled modes the rest group.
  EXPECT_EQ(toString(tile_unzip(
                make_layout(make_tuple(make_tuple(2_c, 5_c), make_tuple(3_c, 7_c), 11_c),
                            make_tuple(make_tuple(1_c, 2_c), make_tuple(10_c, 20_c), 30_c)),
                make_tile(make_layout(make_tuple(4_c, 8_c), make_tuple(1_c, 4_c)),
                          make_layout(2_ct, 1_ct)))),
            "((2,3),(5,7,11)):((1,10),(2,20,30))");
}

// zipped_divide groups the tile modes by the tiler profile, so a tuple tiler keeps its tuple in
// the tile group. tiled_divide unpacks the rest group and flat_divide unpacks both groups.
TEST(TileParity, DivideGrouping) {
  EXPECT_EQ(toString(logical_divide(make_layout(make_tuple(96_c, 5_c), make_tuple(1_c, 256_c)),
                                    make_tile(make_layout(16_ct, 1_ct)))),
            "((16,6),5):((1,16),256)");
  EXPECT_EQ(toString(zipped_divide(make_layout(make_tuple(96_c, 5_c), make_tuple(1_c, 256_c)),
                                   make_tile(make_layout(16_ct, 1_ct)))),
            "((16),(6,5)):((1),(16,256))");
  EXPECT_EQ(toString(tiled_divide(make_layout(make_tuple(96_c, 5_c), make_tuple(1_c, 256_c)),
                                  make_tile(make_layout(16_ct, 1_ct)))),
            "((16),6,5):((1),16,256)");
  // Sources differ: CuTe C++ keeps the rank-1 tile group whole; cutegen unpacks it, as here.
  EXPECT_EQ(toString(flat_divide(make_layout(make_tuple(96_c, 5_c), make_tuple(1_c, 256_c)),
                                 make_tile(make_layout(16_ct, 1_ct)))),
            "(16,6,5):(1,16,256)");
  // Sources differ: CuTe C++ keeps the rank-1 rest group whole; cutegen unpacks it, as here.
  EXPECT_EQ(toString(tiled_divide(make_layout(make_tuple(16_c), make_tuple(1_c)),
                                  make_tile(make_layout(4_ct, 1_ct)))),
            "((4),4):((1),4)");
  EXPECT_EQ(toString(flat_divide(make_layout(make_tuple(16_c), make_tuple(1_c)),
                                 make_tile(make_layout(4_ct, 1_ct)))),
            "(4,4):(1,4)");
  // A rank-1 tiler over a leaf layout gives a rank-1 result: transform_layout rebuilds it through
  // make_layout.
  EXPECT_EQ(toString(logical_divide(make_layout(8_ct, 1_ct), make_tile(make_layout(2_ct, 1_ct)))),
            "((2,4)):((1,2))");
  EXPECT_EQ(toString(zipped_divide(make_layout(8_ct, 1_ct), make_tile(make_layout(2_ct, 1_ct)))),
            "((2),(4)):((1),(2))");
  // Sources differ: cutegen unpacks the rank-1 rest group, as here; CuTe C++ keeps it whole,
  // ((2),(4)):((1),(2)).
  EXPECT_EQ(toString(tiled_divide(make_layout(8_ct, 1_ct), make_tile(make_layout(2_ct, 1_ct)))),
            "((2),4):((1),2)");
  EXPECT_EQ(toString(flat_divide(make_layout(8_ct, 1_ct), make_tile(make_layout(2_ct, 1_ct)))),
            "(2,4):(1,2)");
  EXPECT_EQ(toString(zipped_divide(
                make_layout(make_tuple(64_c, 50_c, 80_c), make_tuple(16000_c, 160_c, 1_c)),
                make_tile(make_layout(32_ct, 1_ct)))),
            "((32),(2,50,80)):((16000),(512000,160,1))");
  EXPECT_EQ(toString(logical_divide(
                make_layout(make_tuple(64_c, 50_c, 80_c), make_tuple(16000_c, 160_c, 1_c)),
                make_tile(make_layout(32_ct, 1_ct), _, make_layout(40_ct, 1_ct)))),
            "((32,2),50,(40,2)):((16000,512000),160,(1,40))");
  EXPECT_EQ(
      toString(logical_divide(make_layout(make_tuple(make_tuple(6_c, 4_c), 8_c),
                                          make_tuple(make_tuple(1_c, 6_c), 24_c)),
                              make_tile(make_tile(make_layout(2_ct, 1_ct), make_layout(2_ct, 1_ct)),
                                        make_layout(4_ct, 1_ct)))),
      "(((2,3),(2,2)),(4,2)):(((1,2),(6,12)),(24,96))");
  EXPECT_EQ(
      toString(zipped_divide(make_layout(make_tuple(make_tuple(6_c, 4_c), 8_c),
                                         make_tuple(make_tuple(1_c, 6_c), 24_c)),
                             make_tile(make_tile(make_layout(2_ct, 1_ct), make_layout(2_ct, 1_ct)),
                                       make_layout(4_ct, 1_ct)))),
      "(((2,2),4),((3,2),2)):(((1,6),24),((2,12),96))");
  EXPECT_EQ(
      toString(tiled_divide(make_layout(make_tuple(make_tuple(6_c, 4_c), 8_c),
                                        make_tuple(make_tuple(1_c, 6_c), 24_c)),
                            make_tile(make_tile(make_layout(2_ct, 1_ct), make_layout(2_ct, 1_ct)),
                                      make_layout(4_ct, 1_ct)))),
      "(((2,2),4),(3,2),2):(((1,6),24),(2,12),96)");
  EXPECT_EQ(
      toString(flat_divide(make_layout(make_tuple(make_tuple(6_c, 4_c), 8_c),
                                       make_tuple(make_tuple(1_c, 6_c), 24_c)),
                           make_tile(make_tile(make_layout(2_ct, 1_ct), make_layout(2_ct, 1_ct)),
                                     make_layout(4_ct, 1_ct)))),
      "((2,2),4,(3,2),2):((1,6),24,(2,12),96)");
  // zip2_by still splits a rank-2 mode under an underscore: (2,(128,16)) gives 2 to the tile group
  // and (128,16) to the rest group.
  EXPECT_EQ(toString(zipped_divide(
                make_layout(
                    make_tuple(make_tuple(2_c, make_tuple(128_c, 16_c)), make_tuple(2_c, 4_c, 2_c)),
                    make_tuple(make_tuple(1_c, make_tuple(2_c, 256_c)),
                               make_tuple(4096_c, 8192_c, 32768_c))),
                make_tile(_, make_tile(make_layout(2_ct, 1_ct), make_layout(1_ct, 1_ct))))),
            "((2,(2,1)),((128,16),(1,4,2))):((1,(4096,8192)),((2,256),(0,8192,32768)))");
  // An underscore tiler leaves the layout whole.
  EXPECT_EQ(toString(zipped_divide(make_layout(make_tuple(64_c, 8_c), make_tuple(1_c, 64_c)), _)),
            "(64,8):(1,64)");
  // zip2_by on a mode the tiler skips.
  EXPECT_EQ(toString(zipped_divide(
                make_layout(make_tuple(64_c, 50_c, 80_c), make_tuple(16000_c, 160_c, 1_c)),
                make_tile(make_layout(32_ct, 1_ct), _, _))),
            "!incompatible-profile:!incompatible-profile");
  EXPECT_EQ(toString(zipped_divide(make_layout(make_tuple(64_c, 32_c), make_tuple(32_c, 1_c)),
                                   make_tile(make_layout(8_ct, 1_ct), make_layout(4_ct, 1_ct),
                                             make_layout(2_ct, 1_ct)))),
            "!tiler-rank-exceeded:!tiler-rank-exceeded");
}

TEST(TileParity, Product) {
  EXPECT_EQ(toString(logical_product(make_layout(make_tuple(2_c, 3_c), make_tuple(1_c, 2_c)),
                                     make_tile(make_layout(2_ct, 1_ct), make_layout(4_ct, 1_ct)))),
            "((2,2),(3,(2,2))):((1,2),(2,(1,6)))");
  EXPECT_EQ(toString(zipped_product(make_layout(make_tuple(2_c, 3_c), make_tuple(3_c, 1_c)),
                                    make_tile(make_layout(2_ct, 1_ct), make_layout(4_ct, 1_ct)))),
            "((2,3),(2,4)):((3,1),(1,3))");
  // Sources differ: CuTe C++ keeps the rank-1 block group whole; cutegen unpacks it, as here.
  EXPECT_EQ(toString(flat_product(make_layout(make_tuple(2_c, 3_c), make_tuple(1_c, 2_c)),
                                  make_tile(make_layout(4_ct, 1_ct)))),
            "(2,4,3):(1,2,2)");
  EXPECT_EQ(toString(logical_product(make_layout(8_ct, 1_ct), make_tile(make_layout(2_ct, 1_ct)))),
            "((8,2)):((1,8))");
  EXPECT_EQ(
      toString(logical_product(make_layout(make_tuple(2_c, 3_c), make_tuple(1_c, 2_c)),
                               make_tile(make_layout(dyn(4, 2), 1_ct), make_layout(4_ct, 1_ct)))),
      "((2,?4{div=2}),(3,(2,2))):((1,2),(2,(1,6)))");
}
