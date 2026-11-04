//===- TiledMMATest.cpp - TiledMMA reference results -----------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// MMAAtom, TiledMMA and ThrMMA of TiledMma.hpp for the CDNA3 MFMA
// f32_16x16x16_f16 atom; the thread-value layouts and partitions are pure
// layout algebra.
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

using namespace core_test;

namespace {

float storage[8192];

MemRef<Runtime> tensor(Lay layout) {
  return make_tensor(Pointer<Runtime>(OpaqueValue::getType<float>(), 32, 16, storage),
                     std::move(layout));
}

int64_t offset(const MemRef<Runtime> &memref) {
  return static_cast<const float *>(memref.data().value()) - storage;
}

/// Four atoms in a 2x2 layout, tiled to 32x64x16: the permutation repeats the
/// atoms once along N.
TiledMMA<Runtime> tiledMMA() {
  return make_tiled_mma(MMAAtom_CDNA3Mfma16x16x16F16(), make_layout(make_tuple(2_c, 2_c)),
                        make_tile(32_ct, 64_ct, 16_ct));
}

} // namespace

TEST(TiledMMAParity, SingleAtom) {
  auto atom = make_tiled_mma(MMAAtom_CDNA3Mfma16x16x16F16());
  EXPECT_EQ(toString(atom.get_layoutA_TV()), "((16,4),(4,(1,1))):((1,64),(16,(0,0)))");
  EXPECT_EQ(toString(atom.get_layoutB_TV()), "((16,4),(4,(1,1))):((1,64),(16,(0,0)))");
  EXPECT_EQ(toString(atom.get_layoutC_TV()), "((16,4),((4,1),(1,1))):((16,4),((1,16),(0,0)))");
}

TEST(TiledMMAParity, ThreadValueLayouts) {
  auto mma = tiledMMA();
  EXPECT_EQ(toString(mma.get_layoutA_TV()), "((16,4,2,2),(4,(1,1))):((1,128,16,0),(32,(0,0)))");
  EXPECT_EQ(toString(mma.get_layoutB_TV()), "((16,4,2,2),(4,(2,1))):((1,256,0,16),(64,(32,0)))");
  EXPECT_EQ(toString(mma.get_layoutC_TV()),
            "((16,8,2),((4,1),(1,2))):((32,4,512),((1,32),(0,1024)))");
  // Without a permutation the tile is one atom per thread group.
  auto plain = make_tiled_mma(MMAAtom_CDNA3Mfma16x16x16F16(), make_layout(make_tuple(2_c, 2_c)));
  EXPECT_EQ(toString(plain.get_layoutC_TV()),
            "((16,8,2),((4,1),(1,1))):((32,4,512),((1,32),(0,0)))");
}

TEST(TiledMMAParity, Sizes) {
  auto mma = tiledMMA();
  EXPECT_EQ(toString(tile_size<0>(mma)), "32");
  EXPECT_EQ(toString(tile_size<1>(mma)), "64");
  EXPECT_EQ(toString(tile_size<2>(mma)), "16");
  EXPECT_EQ(toString(size(mma)), "256");
  EXPECT_EQ(toString(partition_shape_A(mma, make_tuple(128_c, 64_c))), "(4,4,4)");
  EXPECT_EQ(toString(partition_shape_B(mma, make_tuple(128_c, 64_c))), "(4,4,4)");
  EXPECT_EQ(toString(partition_shape_B(mma, make_tuple(128_c, 64_d))), "(4,4,?4)");
  EXPECT_EQ(toString(partition_shape_C(mma, make_tuple(128_c, 128_d))), "((4,1),4,(2,?2))");
}

// Thread 101 is lane 37 of the second atom.
TEST(TiledMMAParity, Partition) {
  auto mma = tiledMMA();
  auto a = tensor(make_layout(make_tuple(64_c, 32_c), make_tuple(32_c, 1_c)));
  auto thrA = mma.get_slice(101_dt).partition_A(a);
  EXPECT_EQ(toString(thrA.layout()), "(4,2,2):(1,1024,16)");
  EXPECT_EQ(offset(thrA), 680);

  auto b = tensor(make_layout(make_tuple(128_c, 32_c), make_tuple(32_c, 1_c)));
  auto thrB = mma.get_slice(101_dt).partition_B(b);
  EXPECT_EQ(toString(thrB.layout()), "(4,4,2):(1,1024,16)");
  EXPECT_EQ(offset(thrB), 168);

  auto c = tensor(make_layout(make_tuple(64_c, 128_d), make_tuple(128_d, 1_c)));
  auto thrC = mma.get_slice(101_dt).partition_C(c);
  EXPECT_EQ(toString(thrC.layout()), "((4,1),2,(2,?2)):((?128,1),?4096{div=32},(32,64))");
  EXPECT_EQ(offset(thrC), 3077);

  EXPECT_EQ(toString(mma.get_slice(0_dt).partition_C(tensor(make_layout(64_ct, 1_ct))).layout()),
            "!tiler-rank-exceeded:!tiler-rank-exceeded");
}

// Tensors whose extents and strides are dynamic multiples; the strides keep their divisibility.
TEST(TiledMMAParity, DynamicPartition) {
  auto mma = tiledMMA();
  EXPECT_EQ(toString(mma.get_thread_slice(101_dt).thr_vmnk()), "(?37,?1,?0,0)");
  auto a = tensor(make_layout(make_tuple(dyn(64, 32), dyn(32, 16)), make_tuple(dyn(32, 16), 1_c)));
  auto thrA = mma.get_thread_slice(101_dt).partition_A(a);
  EXPECT_EQ(toString(thrA.layout()), "(4,?2,?2):(1,?1024{div=512},16)");
  EXPECT_EQ(offset(thrA), 680);

  // A dynamic stride keeps apart the N modes that static strides coalesce to 4:1024.
  auto b = tensor(make_layout(make_tuple(128_c, 32_c), make_tuple(dyn(32, 16), 1_c)));
  EXPECT_EQ(toString(mma.get_slice(101_dt).partition_B(b).layout()),
            "(4,(2,2),2):(1,(?1024{div=512},?2048{div=1024}),16)");

  auto c = tensor(make_layout(make_tuple(64_c, 128_c), make_tuple(1_c, dyn(64, 64))));
  auto thrC = mma.get_slice(101_dt).partition_C(c);
  EXPECT_EQ(toString(thrC.layout()),
            "((4,1),2,(2,2)):((1,?64{div=64}),32,(?2048{div=2048},?4096{div=4096}))");
  EXPECT_EQ(offset(thrC), 344);
  EXPECT_EQ(toString(partition_shape_C(mma, make_tuple(dyn(64, 32), dyn(128, 64)))),
            "((4,1),?2,(2,?2))");
}

// The data of a partitioned coordinate tensor is the coordinate the thread starts at.
TEST(TiledMMAParity, DynamicCoordinateTensor) {
  auto mma = tiledMMA();
  auto c = make_identity_tensor(make_tuple(dyn(64, 32), dyn(128, 64)));
  auto thrC = mma.get_slice(101_dt).partition_C(c);
  EXPECT_EQ(toString(thrC.data()), "(?24{div=4},?5)");
  EXPECT_EQ(toString(thrC.layout()), "((4,1),?2,(2,?2)):((1E0,1E1),32E0,(32E1,64E1))");
}
