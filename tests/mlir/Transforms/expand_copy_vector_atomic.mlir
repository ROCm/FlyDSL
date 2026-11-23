// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-layout-lowering | FileCheck %s

// A packed buffer atomic has a vector value type. Its atom layout covers all
// vector elements, so layout lowering must use the aggregate bit width rather
// than querying the scalar-only Type::getIntOrFloatBitWidth API on the vector.
// CHECK-LABEL: @test_vector_buffer_atomic_copy
// CHECK-NOT: fly.copy(
// CHECK: fly.copy_atom_call
func.func @test_vector_buffer_atomic_copy(
    %atom: !fly.copy_atom<!fly_rocdl.cdna3.buffer_atomic<#fly<atomic_op add>, vector<2xbf16>>, 16>,
    %src: !fly.memref<bf16, register, 2:1>,
    %dst: !fly.memref<bf16, #fly_rocdl.buffer_desc, 2:1>) {
  fly.copy(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_atomic<#fly<atomic_op add>, vector<2xbf16>>, 16>,
         !fly.memref<bf16, register, 2:1>,
         !fly.memref<bf16, #fly_rocdl.buffer_desc, 2:1>) -> ()
  return
}
