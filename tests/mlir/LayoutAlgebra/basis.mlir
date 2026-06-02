// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s | FileCheck %s

// Tests for BasisAttr (scaled-basis / CuTe E<I>) stride leaves flowing through
// the IntTupleBuilder ops: division of a basis coefficient, and a full identity
// layout (1E0, 1E1, ...) through logical_divide. The inferred result types below
// are checked by fly-opt's type inference (a wrong type fails to parse).

// -----

// div divides the scalar coefficient of each basis leaf, keeping its modes:
//   (2E0, 8E1) / (2, 4) = (1E0, 2E1)
// CHECK-LABEL: @test_int_tuple_div_basis
func.func @test_int_tuple_div_basis() -> !fly.int_tuple<(1E0, 2E1)> {
  %a = fly.static : !fly.int_tuple<(2E0, 8E1)>
  %b = fly.static : !fly.int_tuple<(2, 4)>
  // CHECK: fly.int_tuple_div(%{{.*}}, %{{.*}})
  %result = fly.int_tuple_div(%a, %b) : (!fly.int_tuple<(2E0, 8E1)>, !fly.int_tuple<(2, 4)>) -> !fly.int_tuple<(1E0, 2E1)>
  return %result : !fly.int_tuple<(1E0, 2E1)>
}

// -----

// logical_divide partitions a basis-strided identity layout; the algebra walks the
// basis strides (via complement/div) instead of asserting on a non-int leaf.
// CHECK-LABEL: @test_logical_divide_identity
func.func @test_logical_divide_identity() -> !fly.layout<((2, (2, 2)), 4) : ((1E0, (2E0, 1E1)), 2E1)> {
  %s = fly.static : !fly.int_tuple<(4, 8)>
  %id = fly.make_identity_layout(%s) : (!fly.int_tuple<(4, 8)>) -> !fly.layout<(4, 8) : (1E0, 1E1)>
  %ds = fly.static : !fly.int_tuple<(2, 4)>
  %dd = fly.static : !fly.int_tuple<(1, 2)>
  %div = fly.make_layout(%ds, %dd) : (!fly.int_tuple<(2, 4)>, !fly.int_tuple<(1, 2)>) -> !fly.layout<(2, 4) : (1, 2)>
  // CHECK: fly.logical_divide(%{{.*}}, %{{.*}})
  %result = fly.logical_divide(%id, %div) : (!fly.layout<(4, 8) : (1E0, 1E1)>, !fly.layout<(2, 4) : (1, 2)>) -> !fly.layout<((2, (2, 2)), 4) : ((1E0, (2E0, 1E1)), 2E1)>
  return %result : !fly.layout<((2, (2, 2)), 4) : ((1E0, (2E0, 1E1)), 2E1)>
}
