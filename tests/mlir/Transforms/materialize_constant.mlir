// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --canonicalize | FileCheck %s

// Folding a static fly.get_scalar materializes an arith.constant. Nothing in
// the input loads Arith, so the Fly dialect must load it.

// CHECK: %[[C:.*]] = arith.constant 5 : i32
// CHECK: fly.print(%[[C]])
module {
  %t = fly.static : !fly.int_tuple<5>
  %0 = fly.get_scalar(%t) : (!fly.int_tuple<5>) -> i32
  "fly.print"(%0) {format = "x={}"} : (i32) -> ()
}
