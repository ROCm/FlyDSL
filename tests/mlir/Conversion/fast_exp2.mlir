// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-rocdl | FileCheck %s

module {
  gpu.module @kernels {
    // CHECK-LABEL: gpu.func @fast_scalar
    // CHECK: rocdl.exp2
    // CHECK-NOT: math.exp2
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
    // CHECK: %[[E0:.*]] = vector.extract %arg0[0]
    // CHECK: %[[X0:.*]] = rocdl.exp2 %[[E0]]
    // CHECK: %[[I0:.*]] = vector.insert %[[X0]], %{{.*}} [0]
    // CHECK: %[[E1:.*]] = vector.extract %arg0[1]
    // CHECK: %[[X1:.*]] = rocdl.exp2 %[[E1]]
    // CHECK: %[[I1:.*]] = vector.insert %[[X1]], %[[I0]] [1]
    // CHECK: %[[E2:.*]] = vector.extract %arg0[2]
    // CHECK: %[[X2:.*]] = rocdl.exp2 %[[E2]]
    // CHECK: %[[I2:.*]] = vector.insert %[[X2]], %[[I1]] [2]
    // CHECK: %[[E3:.*]] = vector.extract %arg0[3]
    // CHECK: %[[X3:.*]] = rocdl.exp2 %[[E3]]
    // CHECK: vector.insert %[[X3]], %[[I2]] [3]
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
  }

  // CHECK-LABEL: func.func @host_fast_scalar
  // CHECK: math.exp2 %arg0 fastmath<fast> : f32
  // CHECK-NOT: rocdl.exp2
  func.func @host_fast_scalar(%arg0: f32) -> f32 {
    %0 = math.exp2 %arg0 fastmath<fast> : f32
    return %0 : f32
  }
}
