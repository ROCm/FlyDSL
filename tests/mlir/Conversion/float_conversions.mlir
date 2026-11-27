// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --pass-pipeline="builtin.module(gpu.module(convert-rocdl-math-ops))" | FileCheck %s

// Float conversions in convert-rocdl-math-ops, picked for the chip of the #rocdl.target.

// No #rocdl.target: nothing to pick for.
// CHECK-LABEL: gpu.module @no_target
// CHECK: arith.truncf %{{.*}} toward_zero : vector<2xf32> to vector<2xbf16>
gpu.module @no_target {
  gpu.func @bf16_rtz(%x: vector<2xf32>) -> vector<2xbf16> {
    %0 = arith.truncf %x toward_zero : vector<2xf32> to vector<2xbf16>
    gpu.return %0 : vector<2xbf16>
  }
}

// CHECK-LABEL: gpu.module @gfx942
gpu.module @gfx942 [#rocdl.target<chip = "gfx942">] {
  // No v_cvt_pk_bf16_f32: round on the bits, NaN to 0x7fc0 (2143289344), two results per v_perm.
  // CHECK-LABEL: gpu.func @bf16_rne
  // CHECK-COUNT-2: arith.shrui
  // CHECK: arith.cmpf ord
  // CHECK: arith.select %{{.*}}, %{{.*}}, %c2143289344_i32
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.perm"
  // CHECK-NOT: arith.truncf
  gpu.func @bf16_rne(%x: vector<2xf32>) -> vector<2xbf16> {
    %0 = arith.truncf %x : vector<2xf32> to vector<2xbf16>
    gpu.return %0 : vector<2xbf16>
  }

  // nnan drops the NaN check; toward zero keeps the high half as is.
  // CHECK-LABEL: gpu.func @bf16_rtz_nnan
  // CHECK-NOT: arith.cmpf
  // CHECK-NOT: arith.addi
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.perm"
  gpu.func @bf16_rtz_nnan(%x: vector<2xf32>) -> vector<2xbf16> {
    %0 = arith.truncf %x toward_zero fastmath<nnan> : vector<2xf32> to vector<2xbf16>
    gpu.return %0 : vector<2xbf16>
  }

  // Every u8 is a bf16 value: no rounding, no NaN check.
  // CHECK-LABEL: gpu.func @u8_to_bf16
  // CHECK: arith.uitofp %{{.*}} : i8 to f32
  // CHECK-NOT: arith.cmpf
  // CHECK: arith.shrui
  // CHECK-NOT: arith.addi
  gpu.func @u8_to_bf16(%x: i8) -> bf16 {
    %0 = arith.uitofp %x : i8 to bf16
    gpu.return %0 : bf16
  }

  // Also when the exact value went through f32 first, and its conversion was rewritten before.
  // CHECK-LABEL: gpu.func @fp8_via_f32_to_bf16
  // CHECK: rocdl.cvt.pk.f32.fp8
  // CHECK-NOT: arith.cmpf
  // CHECK-NOT: arith.addi
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.perm"
  gpu.func @fp8_via_f32_to_bf16(%x: vector<2xf8E4M3FNUZ>) -> vector<2xbf16> {
    %0 = arith.extf %x : vector<2xf8E4M3FNUZ> to vector<2xf32>
    %1 = arith.truncf %0 : vector<2xf32> to vector<2xbf16>
    gpu.return %1 : vector<2xbf16>
  }

  // An integer is never NaN.
  // CHECK-LABEL: gpu.func @i32_to_bf16
  // CHECK: arith.sitofp %{{.*}} : i32 to f32
  // CHECK: arith.addi
  // CHECK-NOT: arith.cmpf
  gpu.func @i32_to_bf16(%x: i32) -> bf16 {
    %0 = arith.sitofp %x : i32 to bf16
    gpu.return %0 : bf16
  }

  // CHECK-LABEL: gpu.func @f16_rtz
  // CHECK: rocdl.cvt.pkrtz %{{.*}}, %{{.*}} : vector<2xf16>
  // CHECK-NOT: arith.truncf
  gpu.func @f16_rtz(%x: vector<2xf32>) -> vector<2xf16> {
    %0 = arith.truncf %x toward_zero : vector<2xf32> to vector<2xf16>
    gpu.return %0 : vector<2xf16>
  }

  // Other modes: the native rounding to nearest even, moved by one step where the mode differs.
  // CHECK-LABEL: gpu.func @f16_upward
  // CHECK: %[[N:.*]] = arith.truncf %{{.*}} : vector<2xf32> to vector<2xf16>
  // CHECK: %[[B:.*]] = arith.extf %[[N]] : vector<2xf16> to vector<2xf32>
  // CHECK: arith.cmpf olt, %[[B]], %{{.*}} : vector<2xf32>
  // CHECK: arith.addi %{{.*}} : vector<2xi16>
  gpu.func @f16_upward(%x: vector<2xf32>) -> vector<2xf16> {
    %0 = arith.truncf %x upward : vector<2xf32> to vector<2xf16>
    gpu.return %0 : vector<2xf16>
  }

  // CHECK-LABEL: gpu.func @f32_away
  // CHECK: arith.truncf %{{.*}} : f64 to f32
  // CHECK: arith.cmpf oeq
  // CHECK-NOT: toward_zero
  gpu.func @f32_away(%x: f64) -> f32 {
    %0 = arith.truncf %x to_nearest_away : f64 to f32
    gpu.return %0 : f32
  }

  // One native instruction: left to convert-gpu-to-rocdl.
  // CHECK-LABEL: gpu.func @f16_rne
  // CHECK: arith.truncf %{{.*}} : vector<2xf32> to vector<2xf16>
  gpu.func @f16_rne(%x: vector<2xf32>) -> vector<2xf16> {
    %0 = arith.truncf %x : vector<2xf32> to vector<2xf16>
    gpu.return %0 : vector<2xf16>
  }

  // Round to odd in f32, then the native f32 -> f16.
  // CHECK-LABEL: gpu.func @f64_to_f16
  // CHECK: %[[N:.*]] = arith.truncf %{{.*}} : f64 to f32
  // CHECK: arith.extf %[[N]] : f32 to f64
  // CHECK: arith.cmpf ueq
  // CHECK: math.absf
  // CHECK: arith.truncf %{{.*}} : f32 to f16
  gpu.func @f64_to_f16(%x: f64) -> f16 {
    %0 = arith.truncf %x : f64 to f16
    gpu.return %0 : f16
  }

  // afn allows rounding twice.
  // CHECK-LABEL: gpu.func @f64_to_f16_afn
  // CHECK: arith.truncf %{{.*}} : f64 to f32
  // CHECK-NOT: arith.cmpf
  // CHECK: arith.truncf %{{.*}} : f32 to f16
  gpu.func @f64_to_f16_afn(%x: f64) -> f16 {
    %0 = arith.truncf %x fastmath<afn> : f64 to f16
    gpu.return %0 : f16
  }

  // FNUZ fp8 / bf8: four per i32.
  // CHECK-LABEL: gpu.func @f32_to_fp8
  // CHECK: rocdl.cvt.pk.fp8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[false] : i32
  // CHECK: rocdl.cvt.pk.fp8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[true] : i32
  // CHECK: vector.bitcast %{{.*}} : vector<1xi32> to vector<4xf8E4M3FNUZ>
  gpu.func @f32_to_fp8(%x: vector<4xf32>) -> vector<4xf8E4M3FNUZ> {
    %0 = arith.truncf %x : vector<4xf32> to vector<4xf8E4M3FNUZ>
    gpu.return %0 : vector<4xf8E4M3FNUZ>
  }

  // CHECK-LABEL: gpu.func @bf8_to_f32
  // CHECK: vector.bitcast %{{.*}} : vector<4xi8> to vector<1xi32>
  // CHECK: rocdl.cvt.pk.f32.bf8 %{{.*}}[false] : vector<2xf32>
  // CHECK: rocdl.cvt.pk.f32.bf8 %{{.*}}[true] : vector<2xf32>
  gpu.func @bf8_to_f32(%x: vector<4xf8E5M2FNUZ>) -> vector<4xf32> {
    %0 = arith.extf %x : vector<4xf8E5M2FNUZ> to vector<4xf32>
    gpu.return %0 : vector<4xf32>
  }
}

// The target flag `fast` implies nnan and afn.
// CHECK-LABEL: gpu.module @gfx942_fast
gpu.module @gfx942_fast [#rocdl.target<chip = "gfx942", flags = {fast}>] {
  // CHECK-LABEL: gpu.func @bf16_rne
  // CHECK-NOT: arith.cmpf
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.perm"
  gpu.func @bf16_rne(%x: vector<2xf32>) -> vector<2xbf16> {
    %0 = arith.truncf %x : vector<2xf32> to vector<2xbf16>
    gpu.return %0 : vector<2xbf16>
  }
}

// CHECK-LABEL: gpu.module @gfx950
gpu.module @gfx950 [#rocdl.target<chip = "gfx950">] {
  // v_cvt_pk_bf16_f32 rounds to nearest even.
  // CHECK-LABEL: gpu.func @bf16_rne
  // CHECK: arith.truncf %{{.*}} : vector<2xf32> to vector<2xbf16>
  gpu.func @bf16_rne(%x: vector<2xf32>) -> vector<2xbf16> {
    %0 = arith.truncf %x : vector<2xf32> to vector<2xbf16>
    gpu.return %0 : vector<2xbf16>
  }

  // CHECK-LABEL: gpu.func @bf16_rtz
  // CHECK: arith.cmpf ord
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.perm"
  // CHECK-NOT: arith.truncf
  gpu.func @bf16_rtz(%x: vector<2xf32>) -> vector<2xbf16> {
    %0 = arith.truncf %x toward_zero : vector<2xf32> to vector<2xbf16>
    gpu.return %0 : vector<2xbf16>
  }

  // OCP fp8: the same instructions as FNUZ on gfx942.
  // CHECK-LABEL: gpu.func @fp8_to_f32
  // CHECK: rocdl.cvt.pk.f32.fp8 %{{.*}}[false] : vector<2xf32>
  gpu.func @fp8_to_f32(%x: vector<2xf8E4M3FN>) -> vector<2xf32> {
    %0 = arith.extf %x : vector<2xf8E4M3FN> to vector<2xf32>
    gpu.return %0 : vector<2xf32>
  }
}

// CHECK-LABEL: gpu.module @gfx1201
gpu.module @gfx1201 [#rocdl.target<chip = "gfx1201">] {
  // The low word is decoded from the high one.
  // CHECK-LABEL: gpu.func @fp8_to_f32
  // CHECK: arith.shli %{{.*}}, %c16_i32 : i32
  // CHECK: rocdl.cvt.pk.f32.fp8 %{{.*}}[true] : vector<2xf32>
  // CHECK-NOT: [false]
  gpu.func @fp8_to_f32(%x: vector<2xf8E4M3FN>) -> vector<2xf32> {
    %0 = arith.extf %x : vector<2xf8E4M3FN> to vector<2xf32>
    gpu.return %0 : vector<2xf32>
  }
}
