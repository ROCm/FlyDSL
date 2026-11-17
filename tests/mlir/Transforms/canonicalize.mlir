// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-canonicalize | FileCheck %s

// Tests for fly-canonicalize pass:
//   - A static result is rebuilt as a fly.static, whatever its producer
//   - The producer stays when it has side effects
//   - make_shape / make_stride / make_coord become fly.make_int_tuple

// -----

// CHECK-LABEL: @test_static_kept
func.func @test_static_kept() -> !fly.int_tuple<(4, 8)> {
  %0 = fly.static {elems = [4 : i32, 8 : i32]} : !fly.int_tuple<(4, 8)>
  // CHECK-NEXT: %[[S:.*]] = fly.static {{.*}}: !fly.int_tuple<(4,8)>
  // CHECK-NEXT: return %[[S]]
  return %0 : !fly.int_tuple<(4, 8)>
}

// CHECK-LABEL: @test_make_int_tuple_static
func.func @test_make_int_tuple_static() -> !fly.int_tuple<(2, 4, 8)> {
  %0 = fly.make_int_tuple() : () -> !fly.int_tuple<(2, 4, 8)>
  // CHECK-NEXT: %[[S:.*]] = fly.static : !fly.int_tuple<(2,4,8)>
  // CHECK-NEXT: return %[[S]]
  return %0 : !fly.int_tuple<(2, 4, 8)>
}

// CHECK-LABEL: @test_make_layout_static
func.func @test_make_layout_static() -> !fly.layout<(4, 8) : (1, 4)> {
  %s = fly.static : !fly.int_tuple<(4, 8)>
  %d = fly.static : !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  // CHECK-NEXT: %[[L:.*]] = fly.static : !fly.layout<(4,8):(1,4)>
  // CHECK-NEXT: return %[[L]]
  return %layout : !fly.layout<(4, 8) : (1, 4)>
}

// CHECK-LABEL: @test_size_static
func.func @test_size_static() -> !fly.int_tuple<32> {
  %s = fly.static : !fly.int_tuple<(4, 8)>
  %size = fly.size(%s) : (!fly.int_tuple<(4, 8)>) -> !fly.int_tuple<32>
  // CHECK-NEXT: %[[S:.*]] = fly.static : !fly.int_tuple<32>
  // CHECK-NEXT: return %[[S]]
  return %size : !fly.int_tuple<32>
}

// CHECK-LABEL: @test_composition_static
func.func @test_composition_static() -> !fly.layout<(2, 4) : (1, 2)> {
  %s1 = fly.static : !fly.int_tuple<(4, 8)>
  %d1 = fly.static : !fly.int_tuple<(1, 4)>
  %outer = fly.make_layout(%s1, %d1) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %s2 = fly.static : !fly.int_tuple<(2, 4)>
  %d2 = fly.static : !fly.int_tuple<(1, 2)>
  %inner = fly.make_layout(%s2, %d2) : (!fly.int_tuple<(2, 4)>, !fly.int_tuple<(1, 2)>) -> !fly.layout<(2, 4) : (1, 2)>
  %result = fly.composition(%outer, %inner) : (!fly.layout<(4, 8) : (1, 4)>, !fly.layout<(2, 4) : (1, 2)>) -> !fly.layout<(2, 4) : (1, 2)>
  // CHECK-NEXT: %[[L:.*]] = fly.static : !fly.layout<(2,4):(1,2)>
  // CHECK-NEXT: return %[[L]]
  return %result : !fly.layout<(2, 4) : (1, 2)>
}

// CHECK-LABEL: @test_static_atoms
func.func @test_static_atoms() -> (!fly.copy_atom<!fly.universal_copy<32>, 32>, !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>) {
  %copy = fly.make_copy_atom {valBits = 32 : i32} : !fly.copy_atom<!fly.universal_copy<32>, 32>
  %mma = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>
  // CHECK-NOT: fly.make_copy_atom
  // CHECK-NOT: fly.make_mma_atom
  // CHECK-DAG: %[[C:.*]] = fly.static : !fly.copy_atom<!fly.universal_copy<32>, 32>
  // CHECK-DAG: %[[M:.*]] = fly.static : !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>
  // CHECK: return %[[C]], %[[M]]
  return %copy, %mma : !fly.copy_atom<!fly.universal_copy<32>, 32>, !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>
}

// CHECK-LABEL: @test_dynamic_kept
func.func @test_dynamic_kept(%x: i32) -> !fly.int_tuple<(?, 8)> {
  %0 = fly.make_shape(%x) : (i32) -> !fly.int_tuple<(?, 8)>
  // CHECK-NEXT: %[[T:.*]] = fly.make_int_tuple(%{{.*}}) : (i32) -> !fly.int_tuple<(?,8)>
  // CHECK-NEXT: return %[[T]]
  return %0 : !fly.int_tuple<(?, 8)>
}

func.func private @side_effect()
func.func private @make_static() -> !fly.int_tuple<(4, 8)>

// CHECK-LABEL: @test_call_kept
func.func @test_call_kept() -> !fly.int_tuple<(4, 8)> {
  %0 = func.call @make_static() : () -> !fly.int_tuple<(4, 8)>
  // CHECK-NEXT: %[[S:.*]] = fly.static : !fly.int_tuple<(4,8)>
  // CHECK-NEXT: call @make_static()
  // CHECK-NEXT: return %[[S]]
  return %0 : !fly.int_tuple<(4, 8)>
}

// Only the static result is rebuilt; the scf.if and its side effect stay.
// CHECK-LABEL: @test_if_kept
func.func @test_if_kept(%c: i1, %x: i32) -> (!fly.int_tuple<(4, 8)>, i32) {
  %0:2 = scf.if %c -> (!fly.int_tuple<(4, 8)>, i32) {
    func.call @side_effect() : () -> ()
    %a = fly.static : !fly.int_tuple<(4, 8)>
    scf.yield %a, %x : !fly.int_tuple<(4, 8)>, i32
  } else {
    %b = fly.static : !fly.int_tuple<(4, 8)>
    scf.yield %b, %x : !fly.int_tuple<(4, 8)>, i32
  }
  // CHECK-NEXT: %[[S:.*]] = fly.static : !fly.int_tuple<(4,8)>
  // CHECK-NEXT: %[[IF:.*]]:2 = scf.if
  // CHECK-NEXT: func.call @side_effect()
  // CHECK: return %[[S]], %[[IF]]#1
  return %0#0, %0#1 : !fly.int_tuple<(4, 8)>, i32
}
