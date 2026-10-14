// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --split-input-file --verify-diagnostics

func.func @negative_index(%x: i32) {
  // expected-error @+1 {{index must be nonnegative}}
  %pinned = fly.pin_register %x {regClass = #fly.register_class<"amdgcn", "VGPR_32">, index = -1 : i64} : i32
  return
}

// -----

func.func @non_32bit_aligned(%x: i16) {
  // expected-error @+1 {{total bit width must be a multiple of 32}}
  %pinned = fly.pin_register %x {regClass = #fly.register_class<"amdgcn", "VGPR_32">, index = 0 : i64} : i16
  return
}
