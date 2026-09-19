// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-rocdl | FileCheck %s

// ---- 16x16x16 bf16: same-width input ----

// CHECK-LABEL: @test_bf16_16x16x16
// CHECK: llvm.bitcast %{{.*}} : vector<4xbf16> to vector<4xi16>
// CHECK: rocdl.mfma.f32.16x16x16bf16.1k
func.func @test_bf16_16x16x16(
    %a: vector<4xbf16>,
    %b: vector<4xbf16>,
    %c: vector<4xf32>) -> vector<4xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (bf16, bf16) -> f32>>
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (
    !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (bf16, bf16) -> f32>>,
    vector<4xbf16>, vector<4xbf16>, vector<4xf32>) -> vector<4xf32>
  return %res : vector<4xf32>
}

// ---- 16x16x16 bf16: wider input (the bug fix) ----

// CHECK-LABEL: @test_bf16_16x16x16_wide
// CHECK: llvm.bitcast %{{.*}} : vector<8xbf16> to vector<8xi16>
// CHECK: vector.extract_strided_slice
// CHECK: rocdl.mfma.f32.16x16x16bf16.1k
func.func @test_bf16_16x16x16_wide(
    %a: vector<8xbf16>,
    %b: vector<8xbf16>,
    %c: vector<4xf32>) -> vector<4xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (bf16, bf16) -> f32>>
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (
    !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (bf16, bf16) -> f32>>,
    vector<8xbf16>, vector<8xbf16>, vector<4xf32>) -> vector<4xf32>
  return %res : vector<4xf32>
}

// ---- 16x16x16 f16: same-width ----

// CHECK-LABEL: @test_f16_16x16x16
// CHECK: rocdl.mfma.f32.16x16x16f16
func.func @test_f16_16x16x16(
    %a: vector<4xf16>,
    %b: vector<4xf16>,
    %c: vector<4xf32>) -> vector<4xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (
    !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>,
    vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
  return %res : vector<4xf32>
}

// ---- 16x16x16 f16: wider input ----

// CHECK-LABEL: @test_f16_16x16x16_wide
// CHECK: vector.extract_strided_slice
// CHECK: rocdl.mfma.f32.16x16x16f16
func.func @test_f16_16x16x16_wide(
    %a: vector<8xf16>,
    %b: vector<8xf16>,
    %c: vector<4xf32>) -> vector<4xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (
    !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>,
    vector<8xf16>, vector<8xf16>, vector<4xf32>) -> vector<4xf32>
  return %res : vector<4xf32>
}
