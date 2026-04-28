// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-layout-lowering | FileCheck %s

// CHECK-LABEL: @test_mma_make_fragment_with_stages
func.func @test_mma_make_fragment_with_stages(
    %tiled_mma: !fly.tiled_mma<!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>, <(1,1,1):(0,1,2)>>,
    %ptr: !fly.ptr<f16, shared>) -> !fly.memref<f16, register, (4,1,1,2):(1,0,0,4)> {
  %layout = fly.static : !fly.layout<(16,16):(1,16)>
  %input = fly.make_view(%ptr, %layout) : (!fly.ptr<f16, shared>, !fly.layout<(16,16):(1,16)>) -> !fly.memref<f16, shared, (16,16):(1,16)>

  // The optional stages attr appends a stage mode whose dynamic staging stride sorts last,
  // so the final fragment layout places the stage dimension at the highest stride.
  // CHECK: fly.make_layout
  // CHECK-SAME: !fly.int_tuple<(4,1,1,2)>
  // CHECK-SAME: !fly.int_tuple<(1,0,0,4)>
  // CHECK-SAME: !fly.layout<(4,1,1,2):(1,0,0,4)>
  // CHECK: fly.make_ptr() {dictAttrs = {allocaSize = 8 : i64}}
  %frag = fly.mma.make_fragment(b, %tiled_mma, %input, stages = 2) : (!fly.tiled_mma<!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>, <(1,1,1):(0,1,2)>>, !fly.memref<f16, shared, (16,16):(1,16)>) -> !fly.memref<f16, register, (4,1,1,2):(1,0,0,4)>
  return %frag : !fly.memref<f16, register, (4,1,1,2):(1,0,0,4)>
}

// -----

// CHECK-LABEL: @test_mma_make_fragment_without_stages
func.func @test_mma_make_fragment_without_stages(
    %tiled_mma: !fly.tiled_mma<!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>, <(1,1,1):(0,1,2)>>,
    %ptr: !fly.ptr<f16, shared>) -> !fly.memref<f16, register, (4,1,1):(1,0,0)> {
  %layout = fly.static : !fly.layout<(16,16):(1,16)>
  %input = fly.make_view(%ptr, %layout) : (!fly.ptr<f16, shared>, !fly.layout<(16,16):(1,16)>) -> !fly.memref<f16, shared, (16,16):(1,16)>

  // CHECK: fly.make_layout
  // CHECK-SAME: !fly.int_tuple<(4,1,1)>
  // CHECK-SAME: !fly.int_tuple<(1,0,0)>
  // CHECK-SAME: !fly.layout<(4,1,1):(1,0,0)>
  // CHECK: fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}}
  // CHECK-NOT: !fly.layout<(4,1,1,2):(1,0,0,4)>
  %frag = fly.mma.make_fragment(b, %tiled_mma, %input) : (!fly.tiled_mma<!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>, <(1,1,1):(0,1,2)>>, !fly.memref<f16, shared, (16,16):(1,16)>) -> !fly.memref<f16, register, (4,1,1):(1,0,0)>
  return %frag : !fly.memref<f16, register, (4,1,1):(1,0,0)>
}
