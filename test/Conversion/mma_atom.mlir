// RUN: %fly-opt %s --convert-fly-to-rocdl | FileCheck %s

// MMA atom call lowering tests:
//   fly.mma_atom_call -> rocdl.mfma intrinsic
//   Loads A/B as scalars, C as accumulator vector,
//   calls MFMA, stores result back to D

// CHECK-LABEL: @test_mma_atom_call
// CHECK-SAME: (%[[D:.*]]: !llvm.ptr<5>, %[[A:.*]]: !llvm.ptr<5>, %[[B:.*]]: !llvm.ptr<5>, %[[C:.*]]: !llvm.ptr<5>)
func.func @test_mma_atom_call(
    %d: !fly.memref<f32, register, 4:1>,
    %a: !fly.memref<f32, register, 1:1>,
    %b: !fly.memref<f32, register, 1:1>,
    %c: !fly.memref<f32, register, 4:1>) {
  %atom = fly.make_atom : () -> !fly_rocdl.atom.cdna3.mfma<16x16x4, (f32, f32) -> f32>
  // CHECK: %[[A_VAL:.*]] = llvm.load %[[A]] : !llvm.ptr<5> -> f32
  // CHECK: %[[B_VAL:.*]] = llvm.load %[[B]] : !llvm.ptr<5> -> f32
  // CHECK: %[[C_VAL:.*]] = llvm.load %[[C]] : !llvm.ptr<5> -> vector<4xf32>
  // CHECK: %[[RES:.*]] = rocdl.mfma.f32.16x16x4f32 %[[A_VAL]], %[[B_VAL]], %[[C_VAL]]
  // CHECK: llvm.store %[[RES]], %[[D]] : vector<4xf32>, !llvm.ptr<5>
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly_rocdl.atom.cdna3.mfma<16x16x4, (f32, f32) -> f32>, !fly.memref<f32, register, 4:1>, !fly.memref<f32, register, 1:1>, !fly.memref<f32, register, 1:1>, !fly.memref<f32, register, 4:1>) -> ()
  return
}
