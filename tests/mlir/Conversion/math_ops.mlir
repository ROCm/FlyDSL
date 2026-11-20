// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --pass-pipeline="builtin.module(gpu.module(convert-rocdl-math-ops))" | FileCheck %s
// RUN: %fly-opt %s --pass-pipeline="builtin.module(gpu.module(convert-rocdl-math-ops))" | FileCheck %s --check-prefix=VECTOR
// RUN: %fly-opt %s --convert-fly-to-rocdl | FileCheck %s --check-prefix=FLY

module {
  gpu.module @kernels {
    // CHECK-LABEL: gpu.func @fast_scalar
    // CHECK: rocdl.exp2
    // CHECK-NOT: math.exp2
    // FLY-LABEL: gpu.func @fast_scalar
    // FLY: math.exp2 %arg0 fastmath<fast> : f32
    // FLY-NOT: rocdl.exp2
    // FLY: gpu.return
    gpu.func @fast_scalar(%arg0: f32) -> f32 {
      %0 = math.exp2 %arg0 fastmath<fast> : f32
      gpu.return %0 : f32
    }

    // CHECK-LABEL: gpu.func @afn_scalar
    // CHECK: rocdl.exp2
    // CHECK-NOT: math.exp2
    gpu.func @afn_scalar(%arg0: f32) -> f32 {
      %0 = math.exp2 %arg0 fastmath<afn> : f32
      gpu.return %0 : f32
    }

    // CHECK-LABEL: gpu.func @fast_vector
    // VECTOR-LABEL: gpu.func @fast_vector
    // VECTOR-NOT: vector.insert
    // VECTOR-NOT: arith.constant
    // VECTOR: gpu.return
    // CHECK: %[[E0:.*]] = vector.extract %arg0[0]
    // CHECK: %[[X0:.*]] = rocdl.exp2 %[[E0]]
    // CHECK: %[[E1:.*]] = vector.extract %arg0[1]
    // CHECK: %[[X1:.*]] = rocdl.exp2 %[[E1]]
    // CHECK: %[[E2:.*]] = vector.extract %arg0[2]
    // CHECK: %[[X2:.*]] = rocdl.exp2 %[[E2]]
    // CHECK: %[[E3:.*]] = vector.extract %arg0[3]
    // CHECK: %[[X3:.*]] = rocdl.exp2 %[[E3]]
    // CHECK: vector.from_elements %[[X0]], %[[X1]], %[[X2]], %[[X3]] : vector<4xf32>
    // CHECK-NOT: math.exp2
    gpu.func @fast_vector(%arg0: vector<4xf32>) -> vector<4xf32> {
      %0 = math.exp2 %arg0 fastmath<fast> : vector<4xf32>
      gpu.return %0 : vector<4xf32>
    }

    // CHECK-LABEL: gpu.func @strict_scalar
    // CHECK: math.exp2 %arg0 : f32
    // CHECK-NOT: rocdl.exp2
    gpu.func @strict_scalar(%arg0: f32) -> f32 {
      %0 = math.exp2 %arg0 : f32
      gpu.return %0 : f32
    }

    // CHECK-LABEL: gpu.func @non_approx_scalar
    // CHECK: math.exp2 %arg0 fastmath<nnan,ninf> : f32
    // CHECK-NOT: rocdl.exp2
    gpu.func @non_approx_scalar(%arg0: f32) -> f32 {
      %0 = math.exp2 %arg0 fastmath<nnan,ninf> : f32
      gpu.return %0 : f32
    }

    // CHECK-LABEL: gpu.func @fast_f16
    // CHECK: math.exp2 %arg0 fastmath<fast> : f16
    // CHECK-NOT: rocdl.exp2
    gpu.func @fast_f16(%arg0: f16) -> f16 {
      %0 = math.exp2 %arg0 fastmath<fast> : f16
      gpu.return %0 : f16
    }

    // CHECK-LABEL: gpu.func @fast_f64
    // CHECK: math.exp2 %arg0 fastmath<fast> : f64
    // CHECK-NOT: rocdl.exp2
    gpu.func @fast_f64(%arg0: f64) -> f64 {
      %0 = math.exp2 %arg0 fastmath<fast> : f64
      gpu.return %0 : f64
    }

    // log1p(x) = log(w) + c, w = 1 + x, c = x - (w - 1) while w < 2, with no fast-math flags
    // CHECK-LABEL: gpu.func @log1p_afn
    // CHECK-DAG: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
    // CHECK-DAG: %[[TWO:.*]] = arith.constant 2.000000e+00 : f32
    // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK-DAG: %[[LN2:.*]] = arith.constant 0.6931471{{[0-9]*}} : f32
    // CHECK: %[[W:.*]] = arith.addf %arg0, %[[ONE]] : f32
    // CHECK: %[[WM1:.*]] = arith.subf %[[W]], %[[ONE]] : f32
    // CHECK: %[[C:.*]] = arith.subf %arg0, %[[WM1]] : f32
    // CHECK: %[[LT2:.*]] = arith.cmpf olt, %[[W]], %[[TWO]] : f32
    // CHECK: %[[CSEL:.*]] = arith.select %[[LT2]], %[[C]], %[[ZERO]] : f32
    // CHECK: %[[LOG2:.*]] = rocdl.log %[[W]] f32 -> f32
    // CHECK: %[[LOG:.*]] = arith.mulf %[[LOG2]], %[[LN2]] : f32
    // CHECK: arith.addf %[[LOG]], %[[CSEL]] : f32
    // CHECK-NOT: math.log1p
    gpu.func @log1p_afn(%arg0: f32) -> f32 {
      %0 = math.log1p %arg0 fastmath<fast> : f32
      gpu.return %0 : f32
    }

    // f16 keeps the OCML call it gets without this pass
    // CHECK-LABEL: gpu.func @log1p_afn_f16
    // CHECK: llvm.call @__ocml_log1p_f16(%arg0) : (f16) -> f16
    gpu.func @log1p_afn_f16(%arg0: f16) -> f16 {
      %0 = math.log1p %arg0 fastmath<fast> : f16
      gpu.return %0 : f16
    }

    // MathToLLVM's forms are wrong on AMDGPU: these go to OCML, the eight with no AMDGPU lowering
    // whatever the flags
    // CHECK-LABEL: gpu.func @tanh_fast
    // CHECK: llvm.call @__ocml_tanh_f32(%arg0) : (f32) -> f32
    gpu.func @tanh_fast(%arg0: f32) -> f32 {
      %0 = math.tanh %arg0 fastmath<fast> : f32
      gpu.return %0 : f32
    }

    // CHECK-LABEL: gpu.func @log1p_strict
    // CHECK: llvm.call @__ocml_log1p_f32(%arg0) : (f32) -> f32
    gpu.func @log1p_strict(%arg0: f32) -> f32 {
      %0 = math.log1p %arg0 : f32
      gpu.return %0 : f32
    }

    // CHECK-LABEL: gpu.func @expm1_fast
    // CHECK: llvm.call @__ocml_expm1_f32(%arg0) : (f32) -> f32
    gpu.func @expm1_fast(%arg0: f32) -> f32 {
      %0 = math.expm1 %arg0 fastmath<fast> : f32
      gpu.return %0 : f32
    }

    // CHECK-LABEL: gpu.func @rsqrt_strict
    // CHECK: llvm.call @__ocml_rsqrt_f32(%arg0) : (f32) -> f32
    gpu.func @rsqrt_strict(%arg0: f32) -> f32 {
      %0 = math.rsqrt %arg0 : f32
      gpu.return %0 : f32
    }

    // rsqrt with afn keeps the lowering it gets without this pass
    // CHECK-LABEL: gpu.func @rsqrt_fast
    // CHECK: math.rsqrt %arg0 fastmath<fast> : f32
    // CHECK-NOT: llvm.call
    gpu.func @rsqrt_fast(%arg0: f32) -> f32 {
      %0 = math.rsqrt %arg0 fastmath<fast> : f32
      gpu.return %0 : f32
    }

    // CHECK-LABEL: gpu.func @fpowi_runtime
    // CHECK: llvm.call @__ocml_pown_f32(%arg0, %arg1) : (f32, i32) -> f32
    gpu.func @fpowi_runtime(%arg0: f32, %arg1: i32) -> f32 {
      %0 = math.fpowi %arg0, %arg1 : f32, i32
      gpu.return %0 : f32
    }

    // A constant exponent keeps the lowering it gets without this pass
    // CHECK-LABEL: gpu.func @fpowi_constant
    // CHECK: math.fpowi %arg0, %{{.*}} : f32, i32
    // CHECK-NOT: llvm.call
    gpu.func @fpowi_constant(%arg0: f32) -> f32 {
      %c3 = arith.constant 3 : i32
      %0 = math.fpowi %arg0, %c3 : f32, i32
      gpu.return %0 : f32
    }

    // CHECK-LABEL: gpu.func @fpowi_vector
    // CHECK-COUNT-2: llvm.call @__ocml_pown_f32
    gpu.func @fpowi_vector(%arg0: vector<2xf32>) -> vector<2xf32> {
      %c3 = arith.constant dense<3> : vector<2xi32>
      %0 = math.fpowi %arg0, %c3 : vector<2xf32>, vector<2xi32>
      gpu.return %0 : vector<2xf32>
    }
  }

  // CHECK-LABEL: func.func @host_fast_scalar
  // CHECK: math.exp2 %arg0 fastmath<fast> : f32
  // CHECK-NOT: rocdl.exp2
  func.func @host_fast_scalar(%arg0: f32) -> f32 {
    %0 = math.exp2 %arg0 fastmath<fast> : f32
    return %0 : f32
  }
}
