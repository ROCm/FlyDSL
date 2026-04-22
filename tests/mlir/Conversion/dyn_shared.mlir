// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-rocdl | FileCheck %s

// Dynamic shared memory operation lowering tests:
//   - fly.get_dyn_shared -> llvm.mlir.global + llvm.mlir.addressof + llvm.getelementptr
//   - Integration with add_offset, ptr.load, ptr.store
//   - Multiple get_dyn_shared ops reuse same global symbol
//   - Multiple kernels share the same global

// -----

// === Basic lowering ===

// CHECK: gpu.module @basic_module
// CHECK: llvm.mlir.global external @[[SYM0:__dynamic_shared__.*]]() {addr_space = 3 : i32, alignment = 1024 : i64, dso_local} : !llvm.array<0 x i8>
// CHECK-LABEL: gpu.func @test_basic
gpu.module @basic_module {
  gpu.func @test_basic() kernel {
    // CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @[[SYM0]] : !llvm.ptr<3>
    // CHECK: llvm.getelementptr %[[ADDR]][0] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    // CHECK-NOT: fly.get_dyn_shared
    %ptr = fly.get_dyn_shared() : !fly.ptr<i8, shared, align<1024>>
    gpu.return
  }
}

// -----

// === Multiple get_dyn_shared in same kernel reuse the same global ===

// CHECK: gpu.module @reuse_module
// CHECK: llvm.mlir.global external @[[SYM1:__dynamic_shared__.*]]()
// CHECK-LABEL: gpu.func @test_reuse_global
gpu.module @reuse_module {
  gpu.func @test_reuse_global() kernel {
    // CHECK: llvm.mlir.addressof @[[SYM1]]
    // CHECK: llvm.getelementptr
    %ptr0 = fly.get_dyn_shared() : !fly.ptr<i8, shared, align<1024>>
    // CHECK: llvm.mlir.addressof @[[SYM1]]
    // CHECK: llvm.getelementptr
    %ptr1 = fly.get_dyn_shared() : !fly.ptr<i8, shared, align<1024>>
    gpu.return
  }
}

// -----

// === get_dyn_shared + add_offset + ptr.load ===

// CHECK: gpu.module @load_module
// CHECK: llvm.mlir.global external @[[SYM2:__dynamic_shared__.*]]()
// CHECK-LABEL: gpu.func @test_load_from_dyn_shared
gpu.module @load_module {
  gpu.func @test_load_from_dyn_shared(%offset: i32) kernel {
    %shmem = fly.get_dyn_shared() : !fly.ptr<i8, shared, align<1024>>
    %off = fly.make_int_tuple(%offset) : (i32) -> !fly.int_tuple<?>
    // CHECK: llvm.getelementptr {{.*}}[0] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    // CHECK: llvm.getelementptr {{.*}}[%{{.*}}] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %ptr = fly.add_offset(%shmem, %off) : (!fly.ptr<i8, shared, align<1024>>, !fly.int_tuple<?>) -> !fly.ptr<i8, shared>
    // CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> i8
    %val = fly.ptr.load(%ptr) : (!fly.ptr<i8, shared>) -> i8
    gpu.return
  }
}

// -----

// === get_dyn_shared + add_offset + ptr.store ===

// CHECK: gpu.module @store_module
// CHECK: llvm.mlir.global external @[[SYM3:__dynamic_shared__.*]]()
// CHECK-LABEL: gpu.func @test_store_to_dyn_shared
gpu.module @store_module {
  gpu.func @test_store_to_dyn_shared(%offset: i32, %val: i8) kernel {
    %shmem = fly.get_dyn_shared() : !fly.ptr<i8, shared, align<1024>>
    %off = fly.make_int_tuple(%offset) : (i32) -> !fly.int_tuple<?>
    // CHECK: llvm.getelementptr {{.*}}[0]
    // CHECK: llvm.getelementptr {{.*}}[%{{.*}}]
    %ptr = fly.add_offset(%shmem, %off) : (!fly.ptr<i8, shared, align<1024>>, !fly.int_tuple<?>) -> !fly.ptr<i8, shared>
    // CHECK: llvm.store {{.*}}, {{.*}} : i8, !llvm.ptr<3>
    fly.ptr.store(%val, %ptr) : (i8, !fly.ptr<i8, shared>) -> ()
    gpu.return
  }
}

// -----

// === Multiple kernels in same gpu.module share the global ===

// CHECK: gpu.module @multi_kernel_module
// CHECK: llvm.mlir.global external @[[SYM4:__dynamic_shared__.*]]()
// CHECK-NOT: llvm.mlir.global external @__dynamic_shared__
// CHECK-LABEL: gpu.func @kernel_a
gpu.module @multi_kernel_module {
  gpu.func @kernel_a() kernel {
    // CHECK: llvm.mlir.addressof @[[SYM4]]
    %ptr_a = fly.get_dyn_shared() : !fly.ptr<i8, shared, align<1024>>
    gpu.return
  }

  // CHECK-LABEL: gpu.func @kernel_b
  gpu.func @kernel_b() kernel {
    // CHECK: llvm.mlir.addressof @[[SYM4]]
    %ptr_b = fly.get_dyn_shared() : !fly.ptr<i8, shared, align<1024>>
    gpu.return
  }
}
