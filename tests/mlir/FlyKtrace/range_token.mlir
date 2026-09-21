// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

// RUN: %fly-opt %s --convert-fly-ktrace-to-rocdl | FileCheck %s

// range_start's result is declared AnyType, not FlyKtrace_TokenType: the lowering
// replaces it in place with the plain i32 slot index, and a typed result would
// make the generated cast<TypedValue<TokenType>> accessor assert the moment
// range_end's operand became an i32. This test is the regression guard -- before
// the constraint was relaxed, this input crashed the pass.

module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    // CHECK-LABEL: gpu.func @probe
    gpu.func @probe(%buf: !fly.ptr<i8, global>) kernel {
      // The token type is erased: nothing downstream of the pass mentions it.
      // CHECK-NOT: !fly_ktrace.token

      // The prologue's own claim is also a leader-guarded scf.if yielding an i32,
      // and it comes first -- skip past it so SLOT binds to range_start's.
      // CHECK: rocdl.readfirstlane
      //
      // range_start yields its own slot out of the leader scf.if as an i32...
      // CHECK: %[[SLOT:.*]] = scf.if {{.*}} -> (i32)

      // kind is packed into the high byte: START is 3, so 3<<24 | id 1.
      // CHECK: llvm.mlir.constant(50331649 : i32)

      // ...and a wave that did not record yields the UNPAIRED_SLOT sentinel
      // rather than 0, which would alias slot 0 and pair against the cursor.
      // Both the suppressed-store and non-leader paths yield it.
      // CHECK: llvm.mlir.constant(-1 : i32)
      // CHECK: llvm.mlir.constant(-1 : i32)
      %t = fly_ktrace.range_start "phase" -> !fly_ktrace.token

      // END is kind 4 -> 4<<24 | id 1. Both events share event id 1, because
      // they name the same range.
      // CHECK: llvm.mlir.constant(67108865 : i32)

      // The pairing itself: range_start's slot is stored into range_end's
      // start_slot field at byte 28 of the 32-byte record.
      // CHECK: %[[OFF:.*]] = llvm.mlir.constant(28 : i32)
      // CHECK: %[[GEP:.*]] = llvm.getelementptr {{.*}}[%[[OFF]]]
      // CHECK: llvm.store %[[SLOT]], %[[GEP]] : i32, !llvm.ptr<1>
      fly_ktrace.range_end %t, "phase" : !fly_ktrace.token
      gpu.return
    }
  }
}
