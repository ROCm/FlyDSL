// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

// RUN: %fly-opt %s --convert-fly-ktrace-to-rocdl | FileCheck %s

// Event ids must be unique across the whole module, not per kernel. The device
// writes only the integer and the host cannot tell which kernel a record came
// from, so two kernels numbering from 1 would make id 1 ambiguous and render one
// kernel's records under the other's phase name.
//
// The id lands in the record's fifth field, packed as `id | kind << 24`; with
// KIND_MARK == 0 the packed value is the id. Both kernels expand identically
// except for that one constant, so checking that the SECOND kernel materialises
// a 2 is enough -- a per-kernel counter would emit 1 there, and 2 appears
// nowhere else in the expansion.

// The pass also publishes the mapping the host needs to decode a record: the id
// is assigned here, so nothing else can produce this table.
// CHECK: fly_ktrace.event_names = {epilogue = 2 : i32, mainloop = 1 : i32}

module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    // CHECK-LABEL: gpu.func @first
    gpu.func @first(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "mainloop"
      gpu.return
    }

    // CHECK-LABEL: gpu.func @second
    // CHECK: llvm.mlir.constant(2 : i32)
    gpu.func @second(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "epilogue"
      gpu.return
    }
  }
}
