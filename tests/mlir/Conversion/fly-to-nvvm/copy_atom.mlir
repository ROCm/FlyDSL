// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-nvvm | FileCheck %s

// NVVM SM80 copy atom lowering:
//   cp.async (global -> shared) -> nvvm.cp.async.shared.global (cg, 16B)
//   ldmatrix (shared -> register) -> nvvm.ldmatrix m8n8 x4 b16

// CHECK-LABEL: @test_cp_async
func.func @test_cp_async(%s: !fly.memref<f16, global, 8:1>, %d: !fly.memref<f16, shared, 8:1>) {
  %atom = fly.make_copy_atom {valBits = 16 : i32} : !fly.copy_atom<!fly_nvvm.sm80.cp.async<128>, 16>
  // CHECK: nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16, cache = cg : !llvm.ptr<3>, !llvm.ptr<1>
  fly.copy_atom_call(%atom, %s, %d) : (!fly.copy_atom<!fly_nvvm.sm80.cp.async<128>, 16>, !fly.memref<f16, global, 8:1>, !fly.memref<f16, shared, 8:1>) -> ()
  return
}

// CHECK-LABEL: @test_ldmatrix_x4
func.func @test_ldmatrix_x4(%s: !fly.memref<f16, shared, 8:1>, %d: !fly.memref<f16, register, 8:1>) {
  %atom = fly.make_copy_atom {valBits = 16 : i32} : !fly.copy_atom<!fly_nvvm.sm75.ldmatrix<num = 4, trans = false>, 16>
  // CHECK: nvvm.ldmatrix {{.*}} {eltType = #nvvm.ld_st_matrix_elt_type<b16>, layout = #nvvm.mma_layout<row>, num = 4 : i32, shape = #nvvm.ld_st_matrix_shape<m = 8, n = 8>} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
  fly.copy_atom_call(%atom, %s, %d) : (!fly.copy_atom<!fly_nvvm.sm75.ldmatrix<num = 4, trans = false>, 16>, !fly.memref<f16, shared, 8:1>, !fly.memref<f16, register, 8:1>) -> ()
  return
}
