// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --pass-pipeline="builtin.module(gpu.module(convert-rocdl-fastmath-ops))" | FileCheck %s
// RUN: %fly-opt %s --pass-pipeline="builtin.module(gpu.module(convert-rocdl-fastmath-ops))" | FileCheck %s --check-prefix=DECL

// convert-gpu-to-rocdl lowers math.log1p to log(1 + x) and math.expm1 to exp(x) - 1, both 0 for tiny x.
// log1p becomes log(w) + c, w = 1 + x and c = x - (w - 1) while w < 2, with the native log; expm1 calls OCML.

module {
  gpu.module @kernels {
    // CHECK-DAG: llvm.func @__ocml_expm1_f16(f16) -> f16
    // CHECK-DAG: llvm.func @__ocml_expm1_f32(f32) -> f32
    // CHECK-DAG: llvm.func @__ocml_expm1_f64(f64) -> f64
    // DECL: llvm.func @__ocml_expm1_f32
    // DECL-NOT: llvm.func @__ocml_expm1_f32

    // CHECK-LABEL: gpu.func @log1p_f32
    // CHECK-DAG: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
    // CHECK-DAG: %[[TWO:.*]] = arith.constant 2.000000e+00 : f32
    // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK-DAG: %[[LN2:.*]] = arith.constant 0.693147182 : f32
    // CHECK: %[[W:.*]] = arith.addf %arg0, %[[ONE]] : f32
    // CHECK: %[[WM1:.*]] = arith.subf %[[W]], %[[ONE]] : f32
    // CHECK: %[[C:.*]] = arith.subf %arg0, %[[WM1]] : f32
    // CHECK: %[[LT2:.*]] = arith.cmpf olt, %[[W]], %[[TWO]] : f32
    // CHECK: %[[CSEL:.*]] = arith.select %[[LT2]], %[[C]], %[[ZERO]] : f32
    // CHECK: %[[LOG2:.*]] = rocdl.log %[[W]] f32 -> f32
    // CHECK: %[[LOG:.*]] = arith.mulf %[[LOG2]], %[[LN2]] : f32
    // CHECK: %[[R:.*]] = arith.addf %[[LOG]], %[[CSEL]] : f32
    // CHECK: gpu.return %[[R]]
    // CHECK-NOT: math.log1p
    gpu.func @log1p_f32(%arg0: f32) -> f32 {
      %0 = math.log1p %arg0 fastmath<fast> : f32
      gpu.return %0 : f32
    }

    // CHECK-LABEL: gpu.func @log1p_f16
    // CHECK-DAG: %[[LN2:.*]] = arith.constant 6.933590e-01 : f16
    // CHECK: %[[LOG2:.*]] = rocdl.log %{{.*}} f16 -> f16
    // CHECK: arith.mulf %[[LOG2]], %[[LN2]] : f16
    // CHECK-NOT: math.log1p
    gpu.func @log1p_f16(%arg0: f16) -> f16 {
      %0 = math.log1p %arg0 : f16
      gpu.return %0 : f16
    }

    // CHECK-LABEL: gpu.func @log1p_bf16
    // CHECK: %[[X:.*]] = arith.extf %arg0 : bf16 to f32
    // CHECK: arith.addf %[[X]], %{{.*}} : f32
    // CHECK: rocdl.log %{{.*}} f32 -> f32
    // CHECK: arith.truncf %{{.*}} : f32 to bf16
    // CHECK-NOT: math.log1p
    gpu.func @log1p_bf16(%arg0: bf16) -> bf16 {
      %0 = math.log1p %arg0 : bf16
      gpu.return %0 : bf16
    }

    // CHECK-LABEL: gpu.func @log1p_f64
    // CHECK: math.log %{{.*}} : f64
    // CHECK-NOT: rocdl.log
    // CHECK-NOT: math.log1p
    gpu.func @log1p_f64(%arg0: f64) -> f64 {
      %0 = math.log1p %arg0 : f64
      gpu.return %0 : f64
    }

    // CHECK-LABEL: gpu.func @log1p_vector
    // CHECK: %[[W:.*]] = arith.addf %arg0, %{{.*}} : vector<2x2xf32>
    // CHECK: %[[C:.*]] = arith.select %{{.*}} : vector<2x2xi1>, vector<2x2xf32>
    // CHECK: vector.extract %[[W]][0, 0] : f32 from vector<2x2xf32>
    // CHECK-COUNT-4: rocdl.log %{{.*}} f32 -> f32
    // CHECK: %[[LANES:.*]] = vector.from_elements {{.*}} : vector<4xf32>
    // CHECK: %[[LOG:.*]] = vector.shape_cast %[[LANES]] : vector<4xf32> to vector<2x2xf32>
    // CHECK: arith.addf %[[LOG]], %[[C]] : vector<2x2xf32>
    // CHECK-NOT: math.log1p
    gpu.func @log1p_vector(%arg0: vector<2x2xf32>) -> vector<2x2xf32> {
      %0 = math.log1p %arg0 : vector<2x2xf32>
      gpu.return %0 : vector<2x2xf32>
    }

    // CHECK-LABEL: gpu.func @expm1_f32
    // CHECK: %[[R:.*]] = llvm.call @__ocml_expm1_f32(%arg0) : (f32) -> f32
    // CHECK: gpu.return %[[R]]
    // CHECK-NOT: math.expm1
    gpu.func @expm1_f32(%arg0: f32) -> f32 {
      %0 = math.expm1 %arg0 fastmath<fast> : f32
      gpu.return %0 : f32
    }

    // CHECK-LABEL: gpu.func @expm1_f16
    // CHECK: llvm.call @__ocml_expm1_f16(%arg0) : (f16) -> f16
    gpu.func @expm1_f16(%arg0: f16) -> f16 {
      %0 = math.expm1 %arg0 : f16
      gpu.return %0 : f16
    }

    // CHECK-LABEL: gpu.func @expm1_f64
    // CHECK: llvm.call @__ocml_expm1_f64(%arg0) : (f64) -> f64
    gpu.func @expm1_f64(%arg0: f64) -> f64 {
      %0 = math.expm1 %arg0 : f64
      gpu.return %0 : f64
    }

    // CHECK-LABEL: gpu.func @expm1_bf16
    // CHECK: %[[X:.*]] = arith.extf %arg0 : bf16 to f32
    // CHECK: %[[E:.*]] = llvm.call @__ocml_expm1_f32(%[[X]]) : (f32) -> f32
    // CHECK: arith.truncf %[[E]] : f32 to bf16
    gpu.func @expm1_bf16(%arg0: bf16) -> bf16 {
      %0 = math.expm1 %arg0 : bf16
      gpu.return %0 : bf16
    }

    // CHECK-LABEL: gpu.func @expm1_vector
    // CHECK-COUNT-4: llvm.call @__ocml_expm1_f32
    // CHECK: vector.from_elements {{.*}} : vector<4xf32>
    // CHECK-NOT: math.expm1
    gpu.func @expm1_vector(%arg0: vector<4xf32>) -> vector<4xf32> {
      %0 = math.expm1 %arg0 : vector<4xf32>
      gpu.return %0 : vector<4xf32>
    }
  }

  // CHECK-LABEL: func.func @host_log1p
  // CHECK: math.log1p %arg0 : f32
  func.func @host_log1p(%arg0: f32) -> f32 {
    %0 = math.log1p %arg0 : f32
    return %0 : f32
  }
}
