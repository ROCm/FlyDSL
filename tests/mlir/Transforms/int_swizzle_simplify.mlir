// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-int-swizzle-simplify | FileCheck %s

// SwizzleType(M=3, B=3, S=3): mask = 0b111000000 = 448, shift = 3,
// period P = 2^(3+3+3) = 512.

// -----------------------------------------------------------------------------
// (1) Plain swizzle sequence whose input has no peelable structure: untouched.
// -----------------------------------------------------------------------------
// CHECK-LABEL: gpu.func @no_peel
// CHECK:       %[[M:.+]] = arith.constant 448 : i32
// CHECK:       %[[S:.+]] = arith.constant 3   : i32
// CHECK:       %[[A:.+]] = arith.andi %arg0, %[[M]]
// CHECK:       %[[H:.+]] = arith.shrui %[[A]], %[[S]]
// CHECK:       arith.xori %arg0, %[[H]]
// CHECK-NOT:   arith.addi
gpu.module @t1 {
  gpu.func @no_peel(%x: i32) {
    %m = arith.constant 448 : i32
    %s = arith.constant 3   : i32
    %a = arith.andi %x, %m : i32
    %h = arith.shrui %a, %s : i32
    %r = arith.xori %x, %h : i32
    %t = fly.make_int_tuple(%r) : (i32) -> !fly.int_tuple<?>
    gpu.return
  }
}

// -----------------------------------------------------------------------------
// (2) Period-aligned constant addend is peeled out:
//     swizzle(x + 512)  →  swizzle(x) + 512.
// -----------------------------------------------------------------------------
// CHECK-LABEL: gpu.func @peel_const_addend
// CHECK:       %[[D:.+]]   = arith.constant 512 : i32
// CHECK:       %[[M:.+]]   = arith.constant 448 : i32
// CHECK:       %[[S:.+]]   = arith.constant 3   : i32
// CHECK:       %[[AND:.+]] = arith.andi %arg0, %[[M]]
// CHECK:       %[[SHR:.+]] = arith.shrui %[[AND]], %[[S]]
// CHECK:       %[[SW:.+]]  = arith.xori %arg0, %[[SHR]]
// CHECK:       arith.addi %[[SW]], %[[D]]
gpu.module @t2 {
  gpu.func @peel_const_addend(%x: i32) {
    %d = arith.constant 512 : i32
    %y = arith.addi %x, %d : i32
    %m = arith.constant 448 : i32
    %s = arith.constant 3   : i32
    %a = arith.andi %y, %m : i32
    %h = arith.shrui %a, %s : i32
    %r = arith.xori %y, %h : i32
    %t = fly.make_int_tuple(%r) : (i32) -> !fly.int_tuple<?>
    gpu.return
  }
}

// -----------------------------------------------------------------------------
// (3) Whole input divisible by period: swizzle(d) → 0; result is 0 + d = d.
// -----------------------------------------------------------------------------
// CHECK-LABEL: gpu.func @peel_to_zero
// CHECK:       %[[D:.+]]    = arith.constant 1024 : i32
// CHECK:       %[[ZERO:.+]] = arith.constant 0    : i32
// CHECK:       arith.addi %[[ZERO]], %[[D]]
// CHECK-NOT:   arith.xori
gpu.module @t3 {
  gpu.func @peel_to_zero(%x: i32) {
    %d = arith.constant 1024 : i32
    %m = arith.constant 448  : i32
    %s = arith.constant 3    : i32
    %a = arith.andi %d, %m : i32
    %h = arith.shrui %a, %s : i32
    %r = arith.xori %d, %h : i32
    %t = fly.make_int_tuple(%r) : (i32) -> !fly.int_tuple<?>
    gpu.return
  }
}

// -----------------------------------------------------------------------------
// (4) Non-period-aligned addend stays in place (no rewrite).
// -----------------------------------------------------------------------------
// CHECK-LABEL: gpu.func @no_peel_misaligned
// CHECK:       %[[D:.+]] = arith.constant 7 : i32
// CHECK:       arith.addi %arg0, %[[D]]
// CHECK:       arith.xori
// CHECK-NOT:   arith.addi
gpu.module @t4 {
  gpu.func @no_peel_misaligned(%x: i32) {
    %d = arith.constant 7   : i32
    %y = arith.addi %x, %d  : i32
    %m = arith.constant 448 : i32
    %s = arith.constant 3   : i32
    %a = arith.andi %y, %m : i32
    %h = arith.shrui %a, %s : i32
    %r = arith.xori %y, %h : i32
    %t = fly.make_int_tuple(%r) : (i32) -> !fly.int_tuple<?>
    gpu.return
  }
}

// -----------------------------------------------------------------------------
// (5) i64 type works the same.
// -----------------------------------------------------------------------------
// CHECK-LABEL: gpu.func @peel_i64
// CHECK:       %[[D:.+]]   = arith.constant 512 : i64
// CHECK:       %[[M:.+]]   = arith.constant 448 : i64
// CHECK:       %[[S:.+]]   = arith.constant 3   : i64
// CHECK:       %[[AND:.+]] = arith.andi %arg0, %[[M]]
// CHECK:       %[[SHR:.+]] = arith.shrui %[[AND]], %[[S]]
// CHECK:       %[[SW:.+]]  = arith.xori %arg0, %[[SHR]]
// CHECK:       arith.addi %[[SW]], %[[D]]
gpu.module @t5 {
  gpu.func @peel_i64(%x: i64) {
    %d = arith.constant 512 : i64
    %y = arith.addi %x, %d : i64
    %m = arith.constant 448 : i64
    %s = arith.constant 3   : i64
    %a = arith.andi %y, %m : i64
    %h = arith.shrui %a, %s : i64
    %r = arith.xori %y, %h : i64
    %t = fly.make_int_tuple(%r) : (i64) -> !fly.int_tuple<?>
    gpu.return
  }
}
