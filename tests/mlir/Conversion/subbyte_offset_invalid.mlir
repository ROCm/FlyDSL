// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --split-input-file --fly-layout-lowering --convert-fly-to-rocdl --verify-diagnostics

// A static offset into 4- or 6-bit elements that does not land on a byte cannot
// be addressed, through an LLVM pointer or a buffer descriptor.

func.func @fp4_odd(%p: !fly.ptr<f4E2M1FN, global, align<64>>) -> !fly.ptr<f4E2M1FN, global> {
  %o = fly.static : !fly.int_tuple<1>
  // expected-error @+2 {{offset 1 into 4-bit elements does not land on a byte; it must be a multiple of 2}}
  // expected-error @+1 {{failed to legalize operation}}
  %q = fly.add_offset(%p, %o) : (!fly.ptr<f4E2M1FN, global, align<64>>, !fly.int_tuple<1>) -> !fly.ptr<f4E2M1FN, global>
  return %q : !fly.ptr<f4E2M1FN, global>
}

// -----

func.func @slice_odd(%m: !fly.memref<i4, global, (8,4):(1,8), align<64>>) -> !fly.memref<i4, global, 4:8> {
  %c = fly.static : !fly.int_tuple<(1,*)>
  // expected-error @+2 {{offset 1 into 4-bit elements does not land on a byte; it must be a multiple of 2}}
  // expected-error @+1 {{failed to legalize operation}}
  %r = fly.slice(%m, %c) : (!fly.memref<i4, global, (8,4):(1,8), align<64>>, !fly.int_tuple<(1,*)>) -> !fly.memref<i4, global, 4:8>
  return %r : !fly.memref<i4, global, 4:8>
}

// -----

func.func @fp6_not_multiple_of_4(%p: !fly.ptr<f6E2M3FN, shared, align<16>>) -> !fly.ptr<f6E2M3FN, shared> {
  %o = fly.static : !fly.int_tuple<34>
  // expected-error @+2 {{offset 34 into 6-bit elements does not land on a byte; it must be a multiple of 4}}
  // expected-error @+1 {{failed to legalize operation}}
  %q = fly.add_offset(%p, %o) : (!fly.ptr<f6E2M3FN, shared, align<16>>, !fly.int_tuple<34>) -> !fly.ptr<f6E2M3FN, shared>
  return %q : !fly.ptr<f6E2M3FN, shared>
}

// -----

func.func @buffer_odd(%p: !fly.ptr<i4, #fly_rocdl.buffer_desc, align<64>>, %v: vector<2xi4>) {
  %o = fly.static : !fly.int_tuple<3>
  // expected-error @+2 {{offset 3 into 4-bit elements does not land on a byte; it must be a multiple of 2}}
  // expected-error @+1 {{failed to legalize operation}}
  %q = fly.add_offset(%p, %o) : (!fly.ptr<i4, #fly_rocdl.buffer_desc, align<64>>, !fly.int_tuple<3>) -> !fly.ptr<i4, #fly_rocdl.buffer_desc>
  fly.ptr.store(%v, %q) : (vector<2xi4>, !fly.ptr<i4, #fly_rocdl.buffer_desc>) -> ()
  return
}
