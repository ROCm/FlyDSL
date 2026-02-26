// RUN: %fly-opt %s --convert-fly-to-rocdl | FileCheck %s

// Type conversion tests for convert-fly-to-rocdl:
//   - fly.memref -> llvm.ptr with correct address space
//     (global -> 1, shared -> 3, register -> 5)
//   - fly.ptr -> llvm.ptr with correct address space
//   - Supports various element types (f32, f16) and layout shapes

// CHECK-LABEL: @test_func_global_memref
// CHECK-SAME: (%arg0: !llvm.ptr<1>)
func.func @test_func_global_memref(%m: !fly.memref<f32, global, 32:1>) {
  return
}

// CHECK-LABEL: @test_func_shared_memref
// CHECK-SAME: (%arg0: !llvm.ptr<3>)
func.func @test_func_shared_memref(%m: !fly.memref<f32, shared, 32:1>) {
  return
}

// CHECK-LABEL: @test_func_register_memref
// CHECK-SAME: (%arg0: !llvm.ptr<5>)
func.func @test_func_register_memref(%m: !fly.memref<f32, register, 32:1>) {
  return
}

// CHECK-LABEL: @test_func_ptr
// CHECK-SAME: (%arg0: !llvm.ptr<1>)
func.func @test_func_ptr(%p: !fly.ptr<f32, global>) {
  return
}

// CHECK-LABEL: @test_func_shared_ptr
// CHECK-SAME: (%arg0: !llvm.ptr<3>)
func.func @test_func_shared_ptr(%p: !fly.ptr<f32, shared>) {
  return
}

// CHECK-LABEL: @test_func_multi_dim_memref
// CHECK-SAME: (%arg0: !llvm.ptr<1>)
func.func @test_func_multi_dim_memref(%m: !fly.memref<f32, global, (4,8):(1,4)>) {
  return
}

// CHECK-LABEL: @test_func_f16_memref
// CHECK-SAME: (%arg0: !llvm.ptr<3>)
func.func @test_func_f16_memref(%m: !fly.memref<f16, shared, (16,32):(1,16)>) {
  return
}
