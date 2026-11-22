// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-layout-lowering --convert-fly-to-rocdl --canonicalize | FileCheck %s

// 4- and 6-bit elements are packed, but a GEP steps by the alloc size, a whole
// byte for i4 and i6, so an offset into them steps in bytes: offset * bits / 8.
// A dynamic offset is assumed to land on a byte; a static one that does not is
// refused (subbyte_offset_invalid.mlir).

// CHECK-LABEL: @fp4_static
// CHECK: llvm.getelementptr %arg0[16] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
func.func @fp4_static(%p: !fly.ptr<f4E2M1FN, global>) -> !fly.ptr<f4E2M1FN, global> {
  %o = fly.static : !fly.int_tuple<32>
  %q = fly.add_offset(%p, %o) : (!fly.ptr<f4E2M1FN, global>, !fly.int_tuple<32>) -> !fly.ptr<f4E2M1FN, global>
  return %q : !fly.ptr<f4E2M1FN, global>
}

// CHECK-LABEL: @fp6_static
// CHECK: llvm.getelementptr %arg0[24] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
func.func @fp6_static(%p: !fly.ptr<f6E2M3FN, global>) -> !fly.ptr<f6E2M3FN, global> {
  %o = fly.static : !fly.int_tuple<32>
  %q = fly.add_offset(%p, %o) : (!fly.ptr<f6E2M3FN, global>, !fly.int_tuple<32>) -> !fly.ptr<f6E2M3FN, global>
  return %q : !fly.ptr<f6E2M3FN, global>
}

// CHECK-LABEL: @bf6_static_negative
// CHECK: llvm.getelementptr %arg0[-3] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
func.func @bf6_static_negative(%p: !fly.ptr<f6E3M2FN, shared>) -> !fly.ptr<f6E3M2FN, shared> {
  %o = fly.static : !fly.int_tuple<-4>
  %q = fly.add_offset(%p, %o) : (!fly.ptr<f6E3M2FN, shared>, !fly.int_tuple<-4>) -> !fly.ptr<f6E3M2FN, shared>
  return %q : !fly.ptr<f6E3M2FN, shared>
}

// CHECK-LABEL: @i4_dynamic
// CHECK: %[[O:.*]] = llvm.extractvalue %arg1[0]
// CHECK: %[[B:.*]] = arith.shrsi %[[O]], %c1_i64 : i64
// CHECK: llvm.getelementptr %arg0[%[[B]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
func.func @i4_dynamic(%p: !fly.ptr<i4, global, align<64>>, %o: !fly.int_tuple<?{i64}>) -> !fly.ptr<i4, global> {
  %q = fly.add_offset(%p, %o) : (!fly.ptr<i4, global, align<64>>, !fly.int_tuple<?{i64}>) -> !fly.ptr<i4, global>
  return %q : !fly.ptr<i4, global>
}

// CHECK-LABEL: @fp6_dynamic
// CHECK: %[[O:.*]] = llvm.extractvalue %arg1[0]
// CHECK: %[[G:.*]] = arith.shrsi %[[O]], %c2_i32 : i32
// CHECK: %[[B:.*]] = arith.muli %[[G]], %c3_i32 : i32
// CHECK: llvm.getelementptr %arg0[%[[B]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8
func.func @fp6_dynamic(%p: !fly.ptr<f6E2M3FN, global>, %o: !fly.int_tuple<?>) -> !fly.ptr<f6E2M3FN, global> {
  %q = fly.add_offset(%p, %o) : (!fly.ptr<f6E2M3FN, global>, !fly.int_tuple<?>) -> !fly.ptr<f6E2M3FN, global>
  return %q : !fly.ptr<f6E2M3FN, global>
}

// Row 1 of a packed (2, 32) fp4 view starts at byte 16, not 32.
// CHECK-LABEL: @fp4_row
// CHECK: %[[E:.*]] = arith.muli %arg1, %c32_i32 : i32
// CHECK: %[[B:.*]] = arith.shrsi %[[E]], %c1_i32 : i32
// CHECK: %[[P:.*]] = llvm.getelementptr %arg0[%[[B]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8
// CHECK: llvm.load %[[P]] {{.*}} -> vector<32xi4>
func.func @fp4_row(%m: !fly.memref<f4E2M1FN, global, (2,32):(32,1)>, %i: i32) -> vector<32xf4E2M1FN> {
  %c = fly.make_int_tuple(%i) : (i32) -> !fly.int_tuple<(?,*)>
  %r = fly.slice(%m, %c) : (!fly.memref<f4E2M1FN, global, (2,32):(32,1)>, !fly.int_tuple<(?,*)>) -> !fly.memref<f4E2M1FN, global, 32:1>
  %v = fly.memref.load_vec(%r) : (!fly.memref<f4E2M1FN, global, 32:1>) -> vector<32xf4E2M1FN>
  return %v : vector<32xf4E2M1FN>
}

// A byte element keeps its own GEP.
// CHECK-LABEL: @fp8_static
// CHECK-NOT: arith.shrsi
// CHECK: llvm.getelementptr %arg0[32] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
func.func @fp8_static(%p: !fly.ptr<f8E4M3FN, global>) -> !fly.ptr<f8E4M3FN, global> {
  %o = fly.static : !fly.int_tuple<32>
  %q = fly.add_offset(%p, %o) : (!fly.ptr<f8E4M3FN, global>, !fly.int_tuple<32>) -> !fly.ptr<f8E4M3FN, global>
  return %q : !fly.ptr<f8E4M3FN, global>
}

// A buffer pointer carries an element offset and scales it to bytes at the access.
// CHECK-LABEL: @buffer_even
// CHECK: %[[OFF:.*]] = llvm.extractvalue %arg0[1]
// CHECK: %[[ELEM:.*]] = arith.addi %[[OFF]], %c4_i32 : i32
// CHECK: %[[BITS:.*]] = arith.muli %[[ELEM]], %c4_i32 : i32
// CHECK: %[[BYTES:.*]] = arith.divui %[[BITS]], %c8_i32 : i32
// CHECK: rocdl.raw.ptr.buffer.store %arg1, %{{.*}}, %[[BYTES]], %c0_i32, 0 : vector<2xi4>
func.func @buffer_even(%p: !fly.ptr<f4E2M1FN, #fly_rocdl.buffer_desc, align<64>>, %v: vector<2xf4E2M1FN>) {
  %o = fly.static : !fly.int_tuple<4>
  %q = fly.add_offset(%p, %o) : (!fly.ptr<f4E2M1FN, #fly_rocdl.buffer_desc, align<64>>, !fly.int_tuple<4>) -> !fly.ptr<f4E2M1FN, #fly_rocdl.buffer_desc, align<2>>
  fly.ptr.store(%v, %q) : (vector<2xf4E2M1FN>, !fly.ptr<f4E2M1FN, #fly_rocdl.buffer_desc, align<2>>) -> ()
  return
}
