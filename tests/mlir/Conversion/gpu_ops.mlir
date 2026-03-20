// Copyright (c) 2025 FlyDSL Project Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// RUN: %fly-opt %s --convert-fly-to-rocdl | FileCheck %s

// GPU operation lowering tests:
//   - gpu.func signature: fly.memref -> llvm.ptr
//   - gpu.launch_func operands: fly.memref -> llvm.ptr

// -----

// === GPU Function Signature Conversion ===

// CHECK-LABEL: gpu.func @test_gpu_func
// CHECK-SAME: (%arg0: !llvm.ptr<1>)
module {
  gpu.module @test_gpu_module {
    gpu.func @test_gpu_func(%arg0: !fly.memref<f32, global, 32:1>) kernel {
      gpu.return
    }
  }
}

// -----

// === GPU LaunchFunc Operand Conversion ===

// CHECK-LABEL: module attributes {gpu.container_module}
module attributes {gpu.container_module} {
  gpu.module @kernel_mod {
    // CHECK: gpu.func @my_kernel(%arg0: !llvm.ptr<1>)
    gpu.func @my_kernel(%arg0: !fly.memref<f32, global, 32:1>) kernel {
      gpu.return
    }
  }
  // CHECK-LABEL: func.func @test_launch_func
  // CHECK-SAME: (%[[ARG:.*]]: !llvm.ptr<1>)
  func.func @test_launch_func(%arg0: !fly.memref<f32, global, 32:1>) {
    %c1 = arith.constant 1 : index
    // CHECK: gpu.launch_func @kernel_mod::@my_kernel
    // CHECK-SAME: args(%[[ARG]] : !llvm.ptr<1>)
    gpu.launch_func @kernel_mod::@my_kernel
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%arg0 : !fly.memref<f32, global, 32:1>)
    return
  }
}
