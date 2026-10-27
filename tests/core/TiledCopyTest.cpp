//===- TiledCopyTest.cpp - TiledCopy reference results ---------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// CopyAtom, TiledCopy and ThrCopy of TiledCopy.hpp for the CDNA3 buffer copy
// atom; the thread-value layouts and partitions are pure layout algebra.
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

using namespace core_test;

namespace {

float storage[16384];

MemRef<Runtime> tensor(Lay layout) {
  return make_tensor(Pointer<Runtime>(OpaqueValue::getType<float>(), 32, 16, storage),
                     std::move(layout));
}

int64_t offset(const MemRef<Runtime> &memref) {
  return static_cast<const float *>(memref.data().value()) - storage;
}

/// 256 threads in a row-major 32x8 layout, each copying 8 halves with one
/// 128-bit buffer copy.
TiledCopy<Runtime> tiledCopy() {
  return make_tiled_copy(universalCopy(128, 16),
                         make_layout(make_tuple(32_c, 8_c), make_tuple(8_c, 1_c)),
                         make_layout(make_tuple(1_c, 8_c)));
}

/// Four CDNA3 MFMA f32_16x16x16_f16 atoms in a 2x2 layout, tiled to 32x64x16.
TiledMMA<Runtime> tiledMMA() {
  return make_tiled_mma(MMAAtom_CDNA3Mfma16x16x16F16(), make_layout(make_tuple(2_c, 2_c)),
                        make_tile(32_ct, 64_ct, 16_ct));
}

} // namespace

TEST(TiledCopyParity, Atom) {
  // A 128-bit copy of 16-bit values moves 8 values per thread.
  EXPECT_EQ(toString(universalCopy(128, 16).valLayoutSrc()), "(1,8):(1,1)");
}

TEST(TiledCopyParity, ThreadValueLayouts) {
  auto copy = tiledCopy();
  EXPECT_EQ(toString(copy.get_layoutS_TV()), "((8,32),(8,1)):((256,1),(32,0))");
  EXPECT_EQ(toString(copy.get_layoutD_TV()), "((8,32),(8,1)):((256,1),(32,0))");
  EXPECT_EQ(toString(copy.tilerMN().shape()), "(32,64)");
  EXPECT_EQ(toString(size(copy)), "256");
  EXPECT_EQ(toString(copy.tidfrg_S(make_layout(make_tuple(64_c, 64_c), make_tuple(64_c, 1_c)))),
            "((8,32),(8,1),(2,1)):((8,64),(1,0),(2048,0))");
}

TEST(TiledCopyParity, Partition) {
  auto copy = tiledCopy();
  auto s = tensor(make_layout(make_tuple(64_c, 64_c), make_tuple(64_c, 1_c)));
  auto thrS = copy.get_slice(9_dt).partition_S(s);
  EXPECT_EQ(toString(thrS.layout()), "((8,1),2,1):((1,0),2048,0)");
  EXPECT_EQ(offset(thrS), 72);

  auto dst = tensor(make_layout(make_tuple(64_c, 128_d), make_tuple(128_d, 1_c)));
  auto thrD = copy.get_slice(9_dt).partition_D(dst);
  EXPECT_EQ(toString(thrD.layout()), "((8,1),2,?2):((1,0),?4096{div=32},64)");
  EXPECT_EQ(offset(thrD), 136);

  EXPECT_EQ(toString(copy.get_slice(0_dt).partition_S(tensor(make_layout(64_ct, 1_ct))).layout()),
            "!tiler-rank-exceeded:!tiler-rank-exceeded");
}

// Tensors whose extents and strides are dynamic multiples; the strides keep their divisibility.
TEST(TiledCopyParity, DynamicPartition) {
  auto copy = tiledCopy();
  auto rowMajor = make_layout(make_tuple(dyn(64, 32), dyn(128, 64)), make_tuple(dyn(128, 64), 1_c));
  EXPECT_EQ(toString(copy.tidfrg_S(rowMajor)),
            "((8,32),(8,1),(?2,?2)):((8,?128{div=64}),(1,0),(?4096{div=2048},64))");
  auto thrS = copy.get_thread_slice(9_dt).partition_S(tensor(rowMajor));
  EXPECT_EQ(toString(thrS.layout()), "((8,1),?2,?2):((1,0),?4096{div=2048},64)");
  EXPECT_EQ(offset(thrS), 136);

  // A column-major destination with a dynamic leading dimension.
  auto dst = tensor(make_layout(make_tuple(64_c, 128_c), make_tuple(1_c, dyn(64, 64))));
  auto thrD = copy.get_slice(9_dt).partition_D(dst);
  EXPECT_EQ(toString(thrD.layout()), "((8,1),2,2):((?64{div=64},0),32,?4096{div=4096})");
  EXPECT_EQ(offset(thrD), 513);
}

// The data of a partitioned coordinate tensor is the coordinate the thread starts at.
TEST(TiledCopyParity, DynamicCoordinateTensor) {
  auto copy = tiledCopy();
  auto c = make_identity_tensor(make_tuple(dyn(64, 32), dyn(128, 64)));
  auto thrC = copy.get_slice(9_dt).partition_S(c);
  EXPECT_EQ(toString(thrC.data()), "(?1,?8{div=8})");
  EXPECT_EQ(toString(thrC.layout()), "((8,1),?2,?2):((1E1,0),32E0,64E1)");
  auto block = local_tile(c, make_tuple(32_c, 64_c), make_tuple(1_d, 1_d));
  EXPECT_EQ(toString(copy.get_slice(9_dt).partition_S(block).data()), "(?33,?72{div=8})");
}

TEST(TiledCopyParity, FromTiledMMA) {
  auto mma = tiledMMA();
  auto copyA = make_tiled_copy_A(universalCopy(32, 16), mma);
  EXPECT_EQ(toString(copyA.get_layoutS_TV()), "((16,4,2,2),(2,2)):((1,128,16,0),(32,64))");
  EXPECT_EQ(toString(copyA.tilerMN().shape()), "(32,16)");
  auto copyC = make_tiled_copy_C(universalCopy(32, 32), mma);
  EXPECT_EQ(toString(copyC.get_layoutS_TV()), "((16,8,2),(1,(4,2))):((32,4,512),(0,(1,1024)))");
  EXPECT_EQ(toString(copyC.tilerMN().shape()), "(32,64)");
  auto copyB = make_tiled_copy_B(universalCopy(32, 16), mma);
  EXPECT_EQ(toString(copyB.get_layoutS_TV()),
            "((16,4,2,2),(2,(2,2))):((1,256,0,16),(64,(128,32)))");
  EXPECT_EQ(toString(copyB.tilerMN().shape()), "(64,16)");
  // The C atom copy tiles only the values of one atom, so its tiler is a layout per mode.
  auto copyCAtom = make_tiled_copy_C_atom(universalCopy(32, 32), mma);
  EXPECT_EQ(toString(copyCAtom.get_layoutS_TV()), "((16,8,2),(1,1)):((8,1,128),(0,0))");
  EXPECT_EQ(toString(copyCAtom.tilerMN()), "[8:4|32:1]");
}

// make_tiled_copy_S and make_tiled_copy_D tile a new atom by the source or destination
// thread-value layout of another tiled copy.
TEST(TiledCopyParity, FromTiledCopy) {
  auto copyS = make_tiled_copy_S(universalCopy(64, 16), tiledCopy());
  EXPECT_EQ(toString(copyS.get_layoutS_TV()), "((8,32),(4,2)):((256,1),(32,128))");
  EXPECT_EQ(toString(copyS.get_layoutD_TV()), "((8,32),(4,2)):((256,1),(32,128))");
  EXPECT_EQ(toString(copyS.tilerMN().shape()), "(32,64)");
  auto copyD = make_tiled_copy_D(universalCopy(64, 16), tiledCopy());
  EXPECT_EQ(toString(copyD.get_layoutS_TV()), "((8,32),(4,2)):((256,1),(32,128))");
}

// The thread-value layout maps to data addresses, which the data layout maps from coordinates.
TEST(TiledCopyParity, Cotiled) {
  auto copy = make_cotiled_copy(universalCopy(32, 32),
                                make_layout(make_tuple(32_c, 8_c), make_tuple(8_c, 1_c)),
                                make_layout(make_tuple(16_c, 16_c), make_tuple(16_c, 1_c)));
  EXPECT_EQ(toString(copy.get_layoutS_TV()), "((2,16),(1,8)):((16,1),(0,32))");
  EXPECT_EQ(toString(copy.tilerMN()), "[16:1|(2,8):(8,1)]");
}

// retile regroups the values of a (V, rest...) fragment by the atom of this copy.
TEST(TiledCopyParity, Retile) {
  auto copy = make_tiled_copy_impl(
      universalCopy(32, 32),
      make_layout(make_tuple(make_tuple(8_c, 32_c), 8_c), make_tuple(make_tuple(256_c, 1_c), 32_c)),
      make_tuple(32_c, 64_c));
  EXPECT_EQ(toString(copy.retile(make_layout(make_tuple(make_tuple(4_c, 2_c), 2_c, 1_c),
                                             make_tuple(make_tuple(1_c, 4_c), 8_c, 0_c)))),
            "((1,8),2,1):((0,1),8,0)");
  EXPECT_EQ(toString(copy.retile(make_layout(make_tuple(make_tuple(4_c, 2_c), 2_c, 3_d),
                                             make_tuple(make_tuple(1_c, 4_c), 8_c, dyn(16, 16))))),
            "((1,8),2,?3):((0,1),8,?16{div=16})");
  EXPECT_EQ(toString(copy.retile(make_layout(make_tuple(4_c, 2_c), make_tuple(1_c, 4_c)))),
            "!tiler-rank-exceeded:!tiler-rank-exceeded");
}
