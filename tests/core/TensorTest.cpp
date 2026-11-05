//===- TensorTest.cpp - Tensor reference results ---------------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Tensor of Tensor.hpp: the layout a slice or partition leaves and the element
// offset it moves the pointer by.  Coordinate tensors follow make_identity_tensor,
// whose data is the coordinate the partition starts at.
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

using namespace core_test;

namespace {

float storage[4096];

template <class L> MemRef<Runtime> tensor(L layout) {
  return make_tensor(Pointer<Runtime>(OpaqueValue::getType<float>(), 32, 16, storage),
                     std::move(layout));
}

int64_t offset(const MemRef<Runtime> &memref) {
  return static_cast<const float *>(memref.data().value()) - storage;
}

/// (8,?16):(?16,1).
Lay rowMajor() { return make_layout(make_tuple(8_c, 16_d), make_tuple(16_d, 1_c)); }

/// (?64{div=16},?128{div=32}):(?128{div=32},1), an M x K tensor with M a multiple of 16 and K
/// a multiple of 32.
Lay rowMajorMultiples() {
  return make_layout(make_tuple(dyn(64, 16), dyn(128, 32)), make_tuple(dyn(128, 32), 1_c));
}

} // namespace

TEST(TensorParity, Slice) {
  auto t = tensor(rowMajor());
  EXPECT_EQ(toString(slice(make_tuple(_, 3_d), t).layout()), "(8):(?16)");
  EXPECT_EQ(offset(slice(make_tuple(_, 3_d), t)), 3);
  EXPECT_EQ(toString(slice(make_tuple(2_d, _), t).layout()), "(?16):(1)");
  EXPECT_EQ(offset(slice(make_tuple(2_d, _), t)), 32);
  EXPECT_EQ(static_cast<const float *>(t[make_tuple(3_c, 5_c)].value()) - storage, 53);
}

TEST(TensorParity, LocalTile) {
  auto t = tensor(rowMajor());
  EXPECT_EQ(toString(local_tile(t, make_tuple(4_c, 8_c), make_tuple(1_c, 1_c)).layout()),
            "(4,8):(?16,1)");
  EXPECT_EQ(offset(local_tile(t, make_tuple(4_c, 8_c), make_tuple(1_c, 1_c))), 72);
  EXPECT_EQ(toString(local_tile(t, make_tuple(4_c, 8_c), make_tuple(1_d, 1_d)).layout()),
            "(4,8):(?16,1)");
  EXPECT_EQ(offset(local_tile(t, make_tuple(4_c, 8_c), make_tuple(1_d, 1_d))), 72);
  // An underscore keeps the rest mode of the tiles.
  EXPECT_EQ(toString(local_tile(t, make_tuple(4_c, 8_c), make_tuple(1_d, _)).layout()),
            "(4,8,?2):(?16,1,8)");
  EXPECT_EQ(offset(local_tile(t, make_tuple(4_c, 8_c), make_tuple(1_d, _))), 64);

  auto s = tensor(make_layout(make_tuple(8_c, 16_c), make_tuple(16_c, 1_c)));
  EXPECT_EQ(toString(local_tile(s, make_tuple(4_c, 8_c), make_tuple(1_c, 1_c)).layout()),
            "(4,8):(16,1)");

  // Sources differ: Fly unpacks a rank-1 tile group, as here; CuTe C++ keeps it whole,
  // ((16)):((1)) here.
  auto v = tensor(make_layout(make_tuple(96_c), make_tuple(1_c)));
  EXPECT_EQ(toString(local_tile(v, make_tuple(16_c), make_tuple(2_c)).layout()), "(16):(1)");
  EXPECT_EQ(offset(local_tile(v, make_tuple(16_c), make_tuple(2_c))), 32);
  EXPECT_EQ(toString(local_tile(v, make_tuple(16_c), make_tuple(_)).layout()), "(16,6):(1,16)");
  auto m = tensor(make_layout(make_tuple(96_c, 5_c), make_tuple(1_c, 96_c)));
  EXPECT_EQ(toString(local_tile(m, make_tuple(16_c), make_tuple(2_c, _)).layout()),
            "(16,5):(1,96)");
  EXPECT_EQ(offset(local_tile(m, make_tuple(16_c), make_tuple(2_c, _))), 32);

  // A nested tiler mode tiles the matching nested layout mode.
  auto n = tensor(
      make_layout(make_tuple(make_tuple(4_c, 2_c), 8_c), make_tuple(make_tuple(1_c, 4_c), 8_c)));
  EXPECT_EQ(
      toString(local_tile(n, make_tuple(make_tuple(2_c, 2_c), 4_c), make_tuple(1_c, 0_c)).layout()),
      "((2,2),4):((1,4),8)");
  EXPECT_EQ(offset(local_tile(n, make_tuple(make_tuple(2_c, 2_c), 4_c), make_tuple(1_c, 0_c))), 2);
  EXPECT_EQ(offset(local_tile(n, make_tuple(make_tuple(2_c, 2_c), 4_c), make_tuple(1_d, 1_d))), 34);
}

TEST(TensorParity, LocalPartitionAndDivide) {
  auto t = tensor(rowMajor());
  auto threads = make_layout(make_tuple(2_c, 4_c));
  EXPECT_EQ(toString(local_partition(t, threads, 5_dt).layout()), "(4,?4):(?32{div=2},4)");
  EXPECT_EQ(offset(local_partition(t, threads, 5_dt)), 18);
  // Sources differ: Fly unpacks a rank-1 rest group, as here; CuTe C++ keeps it whole,
  // ((3)):((32)) here.
  auto v = tensor(make_layout(make_tuple(96_c), make_tuple(1_c)));
  EXPECT_EQ(toString(local_partition(v, make_layout(make_tuple(32_c)), 3_dt).layout()), "(3):(32)");
  EXPECT_EQ(offset(local_partition(v, make_layout(make_tuple(32_c)), 3_dt)), 3);
  EXPECT_EQ(toString(zipped_divide(t, make_tile(4_ct, 8_ct)).layout()),
            "((4,8),(2,?2)):((?16,1),(?64{div=4},8))");
  EXPECT_EQ(offset(zipped_divide(t, make_tile(4_ct, 8_ct))), 0);
}

// domain_offset moves the data by the offset the coordinate maps to; a composed layout takes the
// offset into the composition instead, leaving the pointer where it is.
TEST(TensorParity, DomainOffset) {
  auto t = tensor(rowMajor());
  EXPECT_EQ(toString(domain_offset(make_tuple(2_d, 3_d), t).layout()), "(8,?16):(?16,1)");
  EXPECT_EQ(offset(domain_offset(make_tuple(2_d, 3_d), t)), 35);
  auto s = tensor(make_composed_layout(Swz(3, 4, 3), 0_ct,
                                       make_layout(make_tuple(8_c, 64_c), make_tuple(64_c, 1_c))));
  EXPECT_EQ(toString(domain_offset(make_tuple(2_c, 3_c), s).layout()),
            "S<3,4,3> o 131 o (8,64):(64,1)");
  EXPECT_EQ(offset(domain_offset(make_tuple(2_c, 3_c), s)), 0);
  EXPECT_EQ(toString(domain_offset(make_tuple(2_d, 3_d), s).layout()),
            "S<3,4,3> o ?131 o (8,64):(64,1)");
}

TEST(TensorParity, CoordinateTensor) {
  // The data is the coordinate the partition starts at.
  auto c = make_identity_tensor(make_tuple(8_c, 16_d));
  auto tile = local_tile(c, make_tuple(4_c, 8_c), make_tuple(1_d, 1_c));
  EXPECT_EQ(toString(tile.data()), "(?4{div=4},8)");
  EXPECT_EQ(toString(tile.layout()), "(4,8):(1E0,1E1)");
  auto part = local_partition(c, make_layout(make_tuple(2_c, 4_c)), 5_dt);
  EXPECT_EQ(toString(part.data()), "(?1,?2)");
  EXPECT_EQ(toString(part.layout()), "(4,?4):(2E0,4E1)");
}

// A block tile at a dynamic block coordinate and its partition by a dynamic thread index; the
// strides keep the divisibility of the leading dimension.
TEST(TensorParity, DynamicTileAndPartition) {
  auto a = tensor(rowMajorMultiples());
  auto tile = local_tile(a, make_tuple(16_c, 32_c), make_tuple(1_d, 1_d));
  EXPECT_EQ(toString(tile.layout()), "(16,32):(?128{div=32},1)");
  EXPECT_EQ(offset(tile), 2080);
  EXPECT_EQ(toString(local_tile(a, make_tuple(16_c, 32_c), make_tuple(_, _)).layout()),
            "(16,32,?4,?4):(?128{div=32},1,?2048{div=512},32)");
  // A multiple of 64 divided by 32 leaves a multiple of 2 tiles.
  auto wide =
      tensor(make_layout(make_tuple(dyn(64, 16), dyn(128, 64)), make_tuple(dyn(128, 64), 1_c)));
  EXPECT_EQ(toString(local_tile(wide, make_tuple(16_c, 32_c), make_tuple(1_d, _)).layout()),
            "(16,32,?4{div=2}):(?128{div=64},1,32)");
  auto part = local_partition(tile, make_layout(make_tuple(4_c, 8_c), make_tuple(8_c, 1_c)), 13_dt);
  EXPECT_EQ(toString(part.layout()), "(4,4):(?512{div=128},8)");
  EXPECT_EQ(offset(part), 2213);

  // A column-major tensor with a dynamic leading dimension.
  auto b = tensor(make_layout(make_tuple(64_c, dyn(32, 8)), make_tuple(1_c, dyn(64, 32))));
  auto bPart = local_partition(local_tile(b, make_tuple(16_c, 8_c), make_tuple(2_d, 3_d)),
                               make_layout(make_tuple(16_c, 4_c)), 37_dt);
  EXPECT_EQ(toString(bPart.layout()), "(1,2):(0,?256{div=128})");
  EXPECT_EQ(offset(bPart), 1701);
}

TEST(TensorParity, DynamicCoordinateTensor) {
  // A dynamic block coordinate scales by the static tile.
  auto c = make_identity_tensor(make_tuple(dyn(64, 16), dyn(128, 32)));
  auto tile = local_tile(c, make_tuple(16_c, 32_c), make_tuple(1_d, _));
  EXPECT_EQ(toString(tile.data()), "(?16{div=16},0)");
  EXPECT_EQ(toString(tile.layout()), "(16,32,?4):(1E0,1E1,32E1)");
  auto part = local_partition(local_tile(c, make_tuple(16_c, 32_c), make_tuple(1_d, 1_d)),
                              make_layout(make_tuple(4_c, 8_c), make_tuple(8_c, 1_c)), 13_dt);
  EXPECT_EQ(toString(part.data()), "(?17,?37)");
  EXPECT_EQ(toString(domain_offset(make_tuple(dyn(16, 16), 32_d), c).data()), "(?16{div=16},?32)");
}
