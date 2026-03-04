// RUN: %fly-opt %s --fly-gpu-to-llvm | FileCheck %s

// fly-gpu-to-llvm pass tests:
//   - asyncObject preserved through gpu.launch_func lowering
//   - No stream created when no async semantics requested
//   - mgpuStreamCreate used as fallback for async token without user stream

module attributes {gpu.container_module} {
  gpu.binary @kernel_bin [#gpu.object<#rocdl.target<chip = "gfx942">, bin = "">]

  // === asyncObject is preserved (the core fix) ===

  // CHECK-LABEL: llvm.func @test_async_object_preserved
  // CHECK-SAME: (%[[ARG0:.*]]: !llvm.ptr, %[[STREAM:.*]]: !llvm.ptr)
  // CHECK-NOT: mgpuStreamCreate
  // CHECK: gpu.launch_func <%[[STREAM]] : !llvm.ptr> @kernel_bin::@my_kernel
  // CHECK-SAME: args(%[[ARG0]] : !llvm.ptr)
  // CHECK-NOT: mgpuStreamDestroy
  func.func @test_async_object_preserved(%arg0: !llvm.ptr, %stream: !llvm.ptr) {
    %c1 = arith.constant 1 : index
    gpu.launch_func <%stream : !llvm.ptr> @kernel_bin::@my_kernel
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%arg0 : !llvm.ptr)
    return
  }

  // === No async: no stream created ===

  // CHECK-LABEL: llvm.func @test_no_async
  // CHECK-NOT: mgpuStreamCreate
  // CHECK: gpu.launch_func
  // CHECK-SAME: @kernel_bin::@my_kernel
  // CHECK-NOT: mgpuStreamSynchronize
  // CHECK-NOT: mgpuStreamDestroy
  func.func @test_no_async(%arg0: !llvm.ptr) {
    %c1 = arith.constant 1 : index
    gpu.launch_func @kernel_bin::@my_kernel
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%arg0 : !llvm.ptr)
    return
  }

  // === Async token without user stream: mgpuStreamCreate fallback ===

  // CHECK-LABEL: llvm.func @test_async_token_fallback
  // CHECK: %[[S:.*]] = llvm.call @mgpuStreamCreate()
  // CHECK: gpu.launch_func <%[[S]] : !llvm.ptr> @kernel_bin::@my_kernel
  // CHECK: llvm.call @mgpuStreamSynchronize(%[[S]])
  // CHECK: llvm.call @mgpuStreamDestroy(%[[S]])
  func.func @test_async_token_fallback(%arg0: !llvm.ptr) {
    %c1 = arith.constant 1 : index
    %token = gpu.launch_func async @kernel_bin::@my_kernel
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%arg0 : !llvm.ptr)
    gpu.wait [%token]
    return
  }

  // === asyncObject with multiple kernel args ===

  // CHECK-LABEL: llvm.func @test_multi_arg_with_stream
  // CHECK-SAME: (%[[A:.*]]: !llvm.ptr, %[[B:.*]]: !llvm.ptr, %[[C:.*]]: !llvm.ptr, %[[STREAM:.*]]: !llvm.ptr)
  // CHECK-NOT: mgpuStreamCreate
  // CHECK: gpu.launch_func <%[[STREAM]] : !llvm.ptr> @kernel_bin::@my_kernel
  // CHECK-SAME: args(%[[A]] : !llvm.ptr, %[[B]] : !llvm.ptr, %[[C]] : !llvm.ptr)
  func.func @test_multi_arg_with_stream(%a: !llvm.ptr, %b: !llvm.ptr, %c: !llvm.ptr, %stream: !llvm.ptr) {
    %c1 = arith.constant 1 : index
    gpu.launch_func <%stream : !llvm.ptr> @kernel_bin::@my_kernel
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%a : !llvm.ptr, %b : !llvm.ptr, %c : !llvm.ptr)
    return
  }

  // === func.func is lowered to llvm.func ===

  // CHECK-LABEL: llvm.func @test_func_lowered
  // CHECK-NOT: func.func
  func.func @test_func_lowered(%arg0: !llvm.ptr, %stream: !llvm.ptr) {
    %c1 = arith.constant 1 : index
    gpu.launch_func <%stream : !llvm.ptr> @kernel_bin::@my_kernel
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%arg0 : !llvm.ptr)
    return
  }
}
