// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-promote-regmem-to-vectorssa | FileCheck %s

// Generic promotion preserves a symbolic class for a different backend.
// CHECK-LABEL: gpu.func @generic_class
// CHECK-NOT: fly.set_register
// CHECK: fly.register_value {{.*}} {bitOffset = 0 : i64, regClass = #fly.register_class<"nvptx64", "Int32Regs">, start = 8 : i64, storageBits = 128 : i64} : vector<8xf16>
// CHECK: fly.register_value {{.*}} {bitOffset = 64 : i64, regClass = #fly.register_class<"nvptx64", "Int32Regs">, start = 8 : i64, storageBits = 128 : i64} : vector<4xf16>
gpu.module @m {
  gpu.func @generic_class(%input: vector<8xf16>, %out: !fly.ptr<f16, global>) kernel {
    %p = fly.make_ptr() {dictAttrs = {allocSize = 8 : i64}} : () -> !fly.ptr<f16, register>
    fly.set_register %p {regClass = #fly.register_class<"nvptx64", "Int32Regs">, start = 8 : i64} : !fly.ptr<f16, register>
    fly.ptr.store(%input, %p) : (vector<8xf16>, !fly.ptr<f16, register>) -> ()
    %offset = fly.make_int_tuple() : () -> !fly.int_tuple<4>
    %q = fly.add_offset(%p, %offset) : (!fly.ptr<f16, register>, !fly.int_tuple<4>) -> !fly.ptr<f16, register>
    %v = fly.ptr.load(%q) : (!fly.ptr<f16, register>) -> vector<4xf16>
    fly.ptr.store(%v, %out) : (vector<4xf16>, !fly.ptr<f16, global>) -> ()
    gpu.return
  }
}
