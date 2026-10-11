// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

// RUN: %fly-opt %s --convert-fly-ktrace-to-rocdl --reconcile-unrealized-casts | FileCheck %s

// A token riding an scf.for iter_arg -- the reason the token form exists, and
// what docs/ktrace_guide.md shows as the headline use: a range whose close
// precedes the open it pairs with, seeded by sentinel_token.
//
// This is the case an in-place replaceAllUsesWith cannot handle. It retypes the
// USE but not the region signature, so the scf.for's result and body block
// argument stay !fly_ktrace.token while the range_end inside reads one, and the
// expansion aborts on a token where it needs an i32. legalizeTokenTypes runs a
// dialect conversion first, which is what rewrites those signatures.
// range_token.mlir does NOT cover this: it is straight-line, all in one block.

module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    // CHECK-LABEL: gpu.func @loop_carried
    gpu.func @loop_carried(%buf: !fly.ptr<i8, global>) kernel {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      // Nothing token-typed survives, and no unrealized cast is left behind for
      // a later pass to trip over.
      // CHECK-NOT: fly_ktrace.token
      // CHECK-NOT: unrealized_conversion_cast

      // The sentinel seeds the iter_arg. It records nothing -- suppression is
      // the frontend's companion i1, not this op -- so it lowers to the
      // UNPAIRED_SLOT constant rather than to a record write.
      // CHECK: %[[SEED:.*]] = llvm.mlir.constant(-1 : i32)
      %init = fly_ktrace.sentinel_token -> !fly_ktrace.token

      // The carried type is now i32, which is the whole point: the conversion
      // rewrote the loop's signature, not just the uses.
      // CHECK: scf.for {{.*}} iter_args(%[[TOK:.*]] = %[[SEED]]) -> (i32)
      %out = scf.for %i = %c0 to %c4 step %c1
             iter_args(%tok = %init) -> (!fly_ktrace.token) {
        // The carried slot is stored as this record's start_slot, at byte 28.
        // CHECK: llvm.mlir.constant(28 : i32)
        // CHECK: llvm.store %[[TOK]]
        fly_ktrace.range_end %tok, "phase" : !fly_ktrace.token
        %next = fly_ktrace.range_start "phase" -> !fly_ktrace.token
        scf.yield %next : !fly_ktrace.token
      }
      fly_ktrace.range_end %out, "phase" : !fly_ktrace.token
      gpu.return
    }
  }
}
