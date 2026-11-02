//===- PointerTest.cpp - Pointer reference results -------------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Pointer of Pointer.hpp: the address after add_offset, and the alignment
// `gcd(offset_bytes, align)`, where a dynamic offset contributes the bytes its
// divisibility guarantees.
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

#include <cstddef>

using namespace core_test;

namespace {

alignas(64) std::byte storage[1024];

Pointer<Runtime> pointer(int32_t bitWidth, int32_t byteAlignment) {
  return Pointer<Runtime>(OpaqueValue::getType<std::byte>(), bitWidth, byteAlignment, storage);
}

} // namespace

TEST(PointerParity, StaticOffset) {
  auto f32 = pointer(32, 16);
  EXPECT_EQ(add_offset(f32, 0_c).value(), storage);
  EXPECT_EQ(add_offset(f32, 0_c).byteAlignment(), 16);
  EXPECT_EQ(add_offset(f32, 3_c).value(), storage + 12);
  EXPECT_EQ(add_offset(f32, 3_c).byteAlignment(), 4);
  EXPECT_EQ(add_offset(f32, 4_c).value(), storage + 16);
  EXPECT_EQ(add_offset(f32, 4_c).byteAlignment(), 16);
  EXPECT_EQ(add_offset(f32, 6_c).byteAlignment(), 8);

  auto f16 = pointer(16, 16);
  EXPECT_EQ(add_offset(f16, 1_c).value(), storage + 2);
  EXPECT_EQ(add_offset(f16, 1_c).byteAlignment(), 2);
  EXPECT_EQ(add_offset(f16, 24_c).byteAlignment(), 16);
}

TEST(PointerParity, DynamicOffset) {
  auto f32 = pointer(32, 16);
  EXPECT_EQ(add_offset(f32, 3_dt).value(), storage + 12);
  EXPECT_EQ(add_offset(f32, 3_dt).byteAlignment(), 4);
  EXPECT_EQ(add_offset(f32, dyn(8, 2)).byteAlignment(), 8);
  EXPECT_EQ(add_offset(f32, dyn(8, 4)).byteAlignment(), 16);
  EXPECT_EQ(add_offset(f32, dyn(16, 8)).byteAlignment(), 16);

  auto f16 = pointer(16, 16);
  EXPECT_EQ(add_offset(f16, 5_dt).value(), storage + 10);
  EXPECT_EQ(add_offset(f16, 5_dt).byteAlignment(), 2);
  EXPECT_EQ(add_offset(f16, dyn(8, 8)).byteAlignment(), 16);

  // The offset cannot raise the alignment above the pointer's own.
  EXPECT_EQ(add_offset(pointer(32, 4), dyn(8, 8)).byteAlignment(), 4);

  auto i8 = pointer(8, 16);
  EXPECT_EQ(add_offset(i8, dyn(6, 2)).value(), storage + 6);
  EXPECT_EQ(add_offset(i8, dyn(6, 2)).byteAlignment(), 2);
}

// A sub-byte element advances by its bit width.
TEST(PointerParity, SubByteOffset) {
  auto i4 = pointer(4, 16);
  EXPECT_EQ(add_offset(i4, dyn(2, 2)).value(), storage + 1);
  EXPECT_EQ(add_offset(i4, dyn(2, 2)).byteAlignment(), 1);
  EXPECT_EQ(add_offset(i4, dyn(8, 8)).value(), storage + 4);
  EXPECT_EQ(add_offset(i4, dyn(8, 8)).byteAlignment(), 4);
}

// A bool stored in bytes advances by the storage width.
TEST(PointerParity, ByteStoredBoolOffset) {
  auto i1 = Pointer<Runtime>(OpaqueValue::getType<std::byte>(), 1, 16, storage, Swz{}, 0,
                             OpaqueValue::getType<void>(), 8);
  EXPECT_EQ(add_offset(i1, dyn(4, 4)).value(), storage + 4);
  EXPECT_EQ(add_offset(i1, dyn(4, 4)).byteAlignment(), 4);
}

// A swizzled pointer offsets the unswizzled address.
TEST(PointerParity, SwizzledOffset) {
  auto f16 = Pointer<Runtime>(OpaqueValue::getType<std::byte>(), 16, 16, storage, Swz(3, 3, 3));
  EXPECT_EQ(add_offset(f16, dyn(64, 8)).value(), storage + 128);
  EXPECT_EQ(add_offset(f16, dyn(64, 8)).byteAlignment(), 16);
  EXPECT_EQ(add_offset(f16, dyn(6, 2)).value(), storage + 12);
  EXPECT_EQ(add_offset(f16, dyn(6, 2)).byteAlignment(), 4);
}
