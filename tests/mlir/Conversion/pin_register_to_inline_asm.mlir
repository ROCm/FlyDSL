// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --split-input-file --convert-fly-to-rocdl | FileCheck %s

// CHECK-LABEL: gpu.func @pin_vgpr_i32
// CHECK: llvm.inline_asm has_side_effects asm_dialect = att "", "={v68},{v68}" %{{.*}} : (i32) -> i32
gpu.module @vgpr_i32 {
  gpu.func @pin_vgpr_i32(%x: i32) kernel {
    %pinned = fly.pin_register %x {regClass = #fly.register_class<"amdgcn", "VGPR_32">, index = 68 : i64} : i32
    gpu.return
  }
}

// -----

// CHECK-LABEL: gpu.func @pin_vgpr_v4i32
// CHECK: llvm.inline_asm has_side_effects asm_dialect = att "", "={v[68:71]},{v[68:71]}" %{{.*}} : (vector<4xi32>) -> vector<4xi32>
gpu.module @vgpr_v4i32 {
  gpu.func @pin_vgpr_v4i32(%x: vector<4xi32>) kernel {
    %pinned = fly.pin_register %x {regClass = #fly.register_class<"amdgcn", "VGPR_32">, index = 68 : i64} : vector<4xi32>
    gpu.return
  }
}

// -----

// CHECK-LABEL: gpu.func @pin_sgpr_i32
// CHECK: llvm.inline_asm has_side_effects asm_dialect = att "", "={s16},{s16}" %{{.*}} : (i32) -> i32
gpu.module @sgpr_i32 {
  gpu.func @pin_sgpr_i32(%x: i32) kernel {
    %pinned = fly.pin_register %x {regClass = #fly.register_class<"amdgcn", "SGPR_32">, index = 16 : i64} : i32
    gpu.return
  }
}

// -----

// CHECK-LABEL: gpu.func @pin_agpr_i32
// CHECK: llvm.inline_asm has_side_effects asm_dialect = att "", "={a0},{a0}" %{{.*}} : (i32) -> i32
gpu.module @agpr_i32 {
  gpu.func @pin_agpr_i32(%x: i32) kernel {
    %pinned = fly.pin_register %x {regClass = #fly.register_class<"amdgcn", "AGPR_32">, index = 0 : i64} : i32
    gpu.return
  }
}

// -----

// CHECK-LABEL: gpu.func @pin_vgpr_f32
// CHECK: llvm.inline_asm has_side_effects asm_dialect = att "", "={v100},{v100}" %{{.*}} : (f32) -> f32
gpu.module @vgpr_f32 {
  gpu.func @pin_vgpr_f32(%x: f32) kernel {
    %pinned = fly.pin_register %x {regClass = #fly.register_class<"amdgcn", "VGPR_32">, index = 100 : i64} : f32
    gpu.return
  }
}
