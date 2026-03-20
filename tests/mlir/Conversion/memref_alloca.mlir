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

// MemRefAlloca lowering tests:
//   fly.memref.alloca -> llvm.alloca with cosize as allocation count

// CHECK-LABEL: @test_memref_alloca
func.func @test_memref_alloca() {
  %s = fly.make_int_tuple() : () -> !fly.int_tuple<(4, 8)>
  %d = fly.make_int_tuple() : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  // Cosize of (4,8):(1,4) = max(4*1, 8*4) = 32
  // CHECK: %[[SIZE:.*]] = arith.constant 32 : i64
  // CHECK: llvm.alloca %[[SIZE]] x f32 : (i64) -> !llvm.ptr<5>
  %mem = fly.memref.alloca(%layout) : (!fly.layout<(4, 8) : (1, 4)>) -> !fly.memref<f32, register, (4, 8) : (1, 4)>
  return
}

// CHECK-LABEL: @test_memref_alloca_1d
func.func @test_memref_alloca_1d() {
  %s = fly.make_int_tuple() : () -> !fly.int_tuple<8>
  %d = fly.make_int_tuple() : () -> !fly.int_tuple<1>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<8>, !fly.int_tuple<1>) -> !fly.layout<8:1>
  // CHECK: %[[SIZE:.*]] = arith.constant 8 : i64
  // CHECK: llvm.alloca %[[SIZE]] x f32 : (i64) -> !llvm.ptr<5>
  %mem = fly.memref.alloca(%layout) : (!fly.layout<8:1>) -> !fly.memref<f32, register, 8:1>
  return
}
