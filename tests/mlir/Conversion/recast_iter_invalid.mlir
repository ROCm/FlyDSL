// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: { not %fly-opt %s -split-input-file 2>&1; } | FileCheck %s

// CHECK: result pointer address space must match source
func.func @bad_recast_address_space(%ptr: !fly.ptr<f32, shared>) {
  %bad = fly.recast_iter(%ptr) : (!fly.ptr<f32, shared>) -> !fly.ptr<f32, global>
  return
}

// -----
// Widening the alignment guarantee is unsound and must be rejected.
// CHECK: result pointer alignment must divide source alignment
func.func @bad_recast_alignment(%ptr: !fly.ptr<f32, shared, align<512>>) {
  %bad = fly.recast_iter(%ptr) : (!fly.ptr<f32, shared, align<512>>) -> !fly.ptr<f32, shared, align<1024>>
  return
}

// -----
// CHECK: result pointer swizzle must match source
func.func @bad_recast_swizzle(%ptr: !fly.ptr<f32, shared, align<1024>, swz<3,3,3>>) {
  %bad = fly.recast_iter(%ptr) : (!fly.ptr<f32, shared, align<1024>, swz<3,3,3>>) -> !fly.ptr<f32, shared, align<1024>>
  return
}
