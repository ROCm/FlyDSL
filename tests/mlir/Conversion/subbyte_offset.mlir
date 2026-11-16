// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-layout-lowering --convert-fly-to-rocdl --canonicalize | FileCheck %s

// A subbyte offset need not identify a whole byte. The result alignment rounds up
// to a whole byte, and the address keeps the element type's own scale.

// CHECK-LABEL: @static_odd
// CHECK: llvm.getelementptr %arg0[1] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i4
func.func @static_odd(%p: !fly.ptr<i4, global, align<64>>) -> !fly.ptr<i4, global> {
  %o = fly.static : !fly.int_tuple<1>
  %q = fly.add_offset(%p, %o) : (!fly.ptr<i4, global, align<64>>, !fly.int_tuple<1>) -> !fly.ptr<i4, global>
  return %q : !fly.ptr<i4, global>
}

// CHECK-LABEL: @dynamic_unaligned
// CHECK: %[[O:.*]] = llvm.extractvalue %arg1[0]
// CHECK: llvm.getelementptr %arg0[%[[O]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i4
func.func @dynamic_unaligned(%p: !fly.ptr<i4, global, align<64>>, %o: !fly.int_tuple<?{i64}>) -> !fly.ptr<i4, global> {
  %q = fly.add_offset(%p, %o) : (!fly.ptr<i4, global, align<64>>, !fly.int_tuple<?{i64}>) -> !fly.ptr<i4, global>
  return %q : !fly.ptr<i4, global>
}

// CHECK-LABEL: @slice_unaligned
// CHECK: llvm.getelementptr %arg0[1] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i4
func.func @slice_unaligned(%m: !fly.memref<i4, global, (8,4):(1,8), align<64>>) -> !fly.memref<i4, global, 4:8> {
  %c = fly.static : !fly.int_tuple<(1,*)>
  %r = fly.slice(%m, %c) : (!fly.memref<i4, global, (8,4):(1,8), align<64>>, !fly.int_tuple<(1,*)>) -> !fly.memref<i4, global, 4:8>
  return %r : !fly.memref<i4, global, 4:8>
}

// A buffer pointer carries an element offset and scales it to bytes at the access.
// CHECK-LABEL: @buffer_odd
// CHECK: %[[OFF:.*]] = llvm.extractvalue %arg0[1]
// CHECK: %[[ELEM:.*]] = arith.addi %[[OFF]], %c3_i32 : i32
// CHECK: %[[BITS:.*]] = arith.muli %[[ELEM]], %c4_i32 : i32
// CHECK: %[[BYTES:.*]] = arith.divui %[[BITS]], %c8_i32 : i32
// CHECK: rocdl.raw.ptr.buffer.store %arg1, %{{.*}}, %[[BYTES]], %c0_i32, 0 : vector<2xi4>
func.func @buffer_odd(%p: !fly.ptr<i4, #fly_rocdl.buffer_desc, align<64>>, %v: vector<2xi4>) {
  %o = fly.static : !fly.int_tuple<3>
  %q = fly.add_offset(%p, %o) : (!fly.ptr<i4, #fly_rocdl.buffer_desc, align<64>>, !fly.int_tuple<3>) -> !fly.ptr<i4, #fly_rocdl.buffer_desc>
  fly.ptr.store(%v, %q) : (vector<2xi4>, !fly.ptr<i4, #fly_rocdl.buffer_desc>) -> ()
  return
}
