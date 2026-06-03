// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s -verify-diagnostics

// -----

func.func @bad_make_layout_profile() {
  %shape = fly.static : !fly.int_tuple<(4, 8)>
  %stride = fly.static : !fly.int_tuple<1>
  // expected-error @+2 {{MakeLayoutOp: expected shape/stride profiles to match for result layout}}
  // expected-error @+1 {{'fly.make_layout' op failed to infer returned types}}
  %layout = fly.make_layout(%shape, %stride) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<1>) -> !fly.layout<(4, 8) : 1>
  return
}

// -----

func.func @bad_composition_static_stride() {
  %outer_shape = fly.static : !fly.int_tuple<(4, 8)>
  %outer_stride = fly.static : !fly.int_tuple<(1, 8)>
  %outer = fly.make_layout(%outer_shape, %outer_stride) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 8)>) -> !fly.layout<(4, 8) : (1, 8)>
  %inner_shape = fly.static : !fly.int_tuple<2>
  %inner_stride = fly.static : !fly.int_tuple<5>
  %inner = fly.make_layout(%inner_shape, %inner_stride) : (!fly.int_tuple<2>, !fly.int_tuple<5>) -> !fly.layout<2 : 5>
  // expected-error @+2 {{CompositionOp: expected inner stride 5 to divide or be smaller than outer shape component 4}}
  // expected-error @+1 {{'fly.composition' op failed to infer returned types}}
  %result = fly.composition(%outer, %inner) : (!fly.layout<(4, 8) : (1, 8)>, !fly.layout<2 : 5>) -> !fly.layout<2 : 5>
  return
}

// -----

func.func @bad_logical_divide_static_size() {
  %layout_shape = fly.static : !fly.int_tuple<16>
  %layout_stride = fly.static : !fly.int_tuple<1>
  %layout = fly.make_layout(%layout_shape, %layout_stride) : (!fly.int_tuple<16>, !fly.int_tuple<1>) -> !fly.layout<16 : 1>
  %divisor_shape = fly.static : !fly.int_tuple<5>
  %divisor_stride = fly.static : !fly.int_tuple<1>
  %divisor = fly.make_layout(%divisor_shape, %divisor_stride) : (!fly.int_tuple<5>, !fly.int_tuple<1>) -> !fly.layout<5 : 1>
  // expected-error @+2 {{LogicalDivideOp: expected static divisor size 5 to divide static layout size 16}}
  // expected-error @+1 {{'fly.logical_divide' op failed to infer returned types}}
  %result = fly.logical_divide(%layout, %divisor) : (!fly.layout<16 : 1>, !fly.layout<5 : 1>) -> !fly.layout<(5, 4) : (1, 5)>
  return
}
