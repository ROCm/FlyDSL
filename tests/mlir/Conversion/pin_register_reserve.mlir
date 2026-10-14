// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --split-input-file --fly-pin-register-reserve | FileCheck %s

// CHECK-LABEL: llvm.func @single_vgpr
// CHECK-SAME: passthrough = {{\[}}{{\[}}"amdgpu-reserved-vgprs", "68"{{\]}}{{\]}}
llvm.func @single_vgpr() attributes {fly.pinned_vgprs = array<i64: 68>} {
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @tuple_vgprs
// CHECK-SAME: passthrough = {{\[}}{{\[}}"amdgpu-reserved-vgprs", "68,69,70,71"{{\]}}{{\]}}
llvm.func @tuple_vgprs() attributes {fly.pinned_vgprs = array<i64: 68, 69, 70, 71>} {
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @sgpr_reserve
// CHECK-SAME: passthrough = {{\[}}{{\[}}"amdgpu-reserved-sgprs", "16"{{\]}}{{\]}}
llvm.func @sgpr_reserve() attributes {fly.pinned_sgprs = array<i64: 16>} {
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @agpr_reserve
// CHECK-SAME: passthrough = {{\[}}{{\[}}"amdgpu-reserved-agprs", "0"{{\]}}{{\]}}
llvm.func @agpr_reserve() attributes {fly.pinned_agprs = array<i64: 0>} {
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @mixed_classes
// CHECK-SAME: "amdgpu-reserved-vgprs", "68"
// CHECK-SAME: "amdgpu-reserved-sgprs", "16"
llvm.func @mixed_classes() attributes {fly.pinned_vgprs = array<i64: 68>, fly.pinned_sgprs = array<i64: 16>} {
  llvm.return
}

// -----

// Duplicates should be deduplicated and sorted.
// CHECK-LABEL: llvm.func @dedup_and_sort
// CHECK-SAME: "amdgpu-reserved-vgprs", "10,20,68"
llvm.func @dedup_and_sort() attributes {fly.pinned_vgprs = array<i64: 68, 10, 20, 68>} {
  llvm.return
}

// -----

// No pinned attrs => no passthrough added.
// CHECK-LABEL: llvm.func @no_pins
// CHECK-NOT: passthrough
llvm.func @no_pins() {
  llvm.return
}
