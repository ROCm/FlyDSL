// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --pass-pipeline="builtin.module(gpu.module(convert-rocdl-fastmath-ops))" | FileCheck %s
// RUN: %fly-opt %s --pass-pipeline="builtin.module(gpu.module(convert-rocdl-fastmath-ops))" | FileCheck %s --check-prefix=VECTOR
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

    // exp(x) = exp2(x * log2(e))
    // CHECK-LABEL: gpu.func @exp
    // CHECK: %[[C:.*]] = arith.constant 1.4426950{{[0-9]*}} : f32
    // CHECK: %[[X:.*]] = arith.mulf %arg0, %[[C]] fastmath<afn> : f32
    // CHECK: rocdl.exp2 %[[X]] f32 -> f32
    // CHECK-NOT: math.exp
    gpu.func @exp(%arg0: f32) -> f32 {
      %0 = math.exp %arg0 fastmath<afn> : f32
      gpu.return %0 : f32
    }

    // CHECK-LABEL: gpu.func @log2
    // CHECK: rocdl.log %arg0 f32 -> f32
    // CHECK-NOT: math.log2
    gpu.func @log2(%arg0: f32) -> f32 {
      %0 = math.log2 %arg0 fastmath<afn> : f32
      gpu.return %0 : f32
    }

    // log(x) = log2(x) * ln(2)
    // CHECK-LABEL: gpu.func @log
    // CHECK: %[[C:.*]] = arith.constant 0.6931471{{[0-9]*}} : f32
    // CHECK: %[[L:.*]] = rocdl.log %arg0 f32 -> f32
    // CHECK: arith.mulf %[[L]], %[[C]] fastmath<afn> : f32
    // CHECK-NOT: math.log
    gpu.func @log(%arg0: f32) -> f32 {
      %0 = math.log %arg0 fastmath<afn> : f32
      gpu.return %0 : f32
    }

    // log10(x) = log2(x) * log10(2), also in f16
    // CHECK-LABEL: gpu.func @log10_f16
    // CHECK: %[[C:.*]] = arith.constant 3.01{{[0-9]*}}e-01 : f16
    // CHECK: %[[L:.*]] = rocdl.log %arg0 f16 -> f16
    // CHECK: arith.mulf %[[L]], %[[C]] fastmath<fast> : f16
    // CHECK-NOT: math.log10
    gpu.func @log10_f16(%arg0: f16) -> f16 {
      %0 = math.log10 %arg0 fastmath<fast> : f16
      gpu.return %0 : f16
    }

    // sin / cos take the operand in revolutions: x / (2 pi)
    // CHECK-LABEL: gpu.func @sin
    // CHECK: %[[C:.*]] = arith.constant 0.1591549{{[0-9]*}} : f32
    // CHECK: %[[R:.*]] = arith.mulf %arg0, %[[C]] fastmath<afn> : f32
    // CHECK: rocdl.sin %[[R]] f32 -> f32
    // CHECK-NOT: math.sin
    gpu.func @sin(%arg0: f32) -> f32 {
      %0 = math.sin %arg0 fastmath<afn> : f32
      gpu.return %0 : f32
    }

    // CHECK-LABEL: gpu.func @cos
    // CHECK: %[[C:.*]] = arith.constant 0.1591549{{[0-9]*}} : f32
    // CHECK: %[[R:.*]] = arith.mulf %arg0, %[[C]] fastmath<afn> : f32
    // CHECK: rocdl.cos %[[R]] f32 -> f32
    // CHECK-NOT: math.cos
    gpu.func @cos(%arg0: f32) -> f32 {
      %0 = math.cos %arg0 fastmath<afn> : f32
      gpu.return %0 : f32
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
