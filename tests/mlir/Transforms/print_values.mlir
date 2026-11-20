// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-layout-lowering | FileCheck %s

// fly.print accepts every Fly value: layouts print their dynamic leaves, a memref
// prints its layout, a coordinate tensor its base and layout, and other values
// (pointers, tiles, atoms) their static type. A composed layout prints as its type:
// a composed inner continues the chain and a composed outer is bracketed.

// CHECK-LABEL: gpu.func @print_values
// CHECK: gpu.printf "m=(4,%d):(1,4) x=%d\0A"
// CHECK: gpu.printf "!fly.ptr<f32, global> S<2,3,3> o 0 o (8,%d):(%d,1) %ld o (%d,4):(1,%ld)\0A"
// CHECK: gpu.printf "S<2,3,3> o 0 o 64:1 o 1 o [32:1 o %d o 8:4]\0A"
gpu.module @m {
  gpu.func @print_values(%m: !fly.memref<f32, global, (4,?):(1,4)>, %x: i32, %p: !fly.ptr<f32, global>,
                         %c: !fly.composed_layout<S<2,3,3> o 0 o (8,?) : (?,1)>,
                         %t: !fly.coord_tensor<?{i64},(?,4):(1,?{i64})>,
                         %n: !fly.composed_layout<S<2,3,3> o 0 o 64:1 o 1 o [32:1 o ? o 8:4]>) kernel {
    "fly.print"(%m, %x) {format = "m={} x={}"} : (!fly.memref<f32, global, (4,?):(1,4)>, i32) -> ()
    "fly.print"(%p, %c, %t) {format = ""} : (!fly.ptr<f32, global>, !fly.composed_layout<S<2,3,3> o 0 o (8,?) : (?,1)>, !fly.coord_tensor<?{i64},(?,4):(1,?{i64})>) -> ()
    "fly.print"(%n) {format = ""} : (!fly.composed_layout<S<2,3,3> o 0 o 64:1 o 1 o [32:1 o ? o 8:4]>) -> ()
    gpu.return
  }
}
