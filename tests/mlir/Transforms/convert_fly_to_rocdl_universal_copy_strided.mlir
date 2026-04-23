// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-convert-atom-call-to-ssa-form --convert-fly-to-rocdl | FileCheck %s

gpu.module @bug_strided_universal_copy {

// CHECK-LABEL: gpu.func @load_strided_global_into_register(
// CHECK-SAME:     %[[ARG0:.*]]: !llvm.ptr<1>
// CHECK:      %[[REG:.*]] = llvm.alloca %{{.*}} x f16 : (i64) -> !llvm.ptr<5>
// CHECK:      %[[V:.*]] = llvm.load %[[ARG0]] : !llvm.ptr<1> -> vector<4xf16>
// CHECK-NEXT: llvm.store %[[V]], %[[REG]] : vector<4xf16>, !llvm.ptr<5>
  gpu.func @load_strided_global_into_register(%src: !fly.ptr<f16,  global>) kernel {
    %shape4  = fly.make_int_tuple() : () -> !fly.int_tuple<4>
    %stride1 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %stride8 = fly.make_int_tuple() : () -> !fly.int_tuple<8>

    %src_layout = fly.make_layout(%shape4, %stride8)
        : (!fly.int_tuple<4>, !fly.int_tuple<8>) -> !fly.layout<4:8>
    %reg_layout = fly.make_layout(%shape4, %stride1)
        : (!fly.int_tuple<4>, !fly.int_tuple<1>) -> !fly.layout<4:1>

    %src_view = fly.make_view(%src, %src_layout)
        : (!fly.ptr<f16,  global>, !fly.layout<4:8>) -> !fly.memref<f16,  global, 4:8>

    %copy = fly.make_copy_atom {valBits = 16 : i32}
        : !fly.copy_atom<!fly.universal_copy<64>, 16>

    %reg_ptr  = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}}
        : () -> !fly.ptr<f16,  register>
    %reg_view = fly.make_view(%reg_ptr, %reg_layout)
        : (!fly.ptr<f16,  register>, !fly.layout<4:1>) -> !fly.memref<f16,  register, 4:1>

    fly.copy_atom_call(%copy, %src_view, %reg_view)
        : (!fly.copy_atom<!fly.universal_copy<64>, 16>,
           !fly.memref<f16,  global, 4:8>,
           !fly.memref<f16,  register, 4:1>) -> ()
    gpu.return
  }

// CHECK-LABEL: gpu.func @store_register_into_strided_global(
// CHECK-SAME:     %[[ARG0:.*]]: !llvm.ptr<1>
// CHECK:      %[[REG:.*]] = llvm.alloca %{{.*}} x f16 : (i64) -> !llvm.ptr<5>
// CHECK:      %[[V:.*]] = llvm.load %[[REG]] : !llvm.ptr<5> -> vector<4xf16>
// CHECK-NEXT: llvm.store %[[V]], %[[ARG0]] : vector<4xf16>, !llvm.ptr<1>
  gpu.func @store_register_into_strided_global(%dst: !fly.ptr<f16,  global>) kernel {
    %shape4  = fly.make_int_tuple() : () -> !fly.int_tuple<4>
    %stride1 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %stride8 = fly.make_int_tuple() : () -> !fly.int_tuple<8>

    %dst_layout = fly.make_layout(%shape4, %stride8)
        : (!fly.int_tuple<4>, !fly.int_tuple<8>) -> !fly.layout<4:8>
    %reg_layout = fly.make_layout(%shape4, %stride1)
        : (!fly.int_tuple<4>, !fly.int_tuple<1>) -> !fly.layout<4:1>

    %dst_view = fly.make_view(%dst, %dst_layout)
        : (!fly.ptr<f16,  global>, !fly.layout<4:8>) -> !fly.memref<f16,  global, 4:8>

    %copy = fly.make_copy_atom {valBits = 16 : i32}
        : !fly.copy_atom<!fly.universal_copy<64>, 16>

    %reg_ptr  = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}}
        : () -> !fly.ptr<f16,  register>
    %reg_view = fly.make_view(%reg_ptr, %reg_layout)
        : (!fly.ptr<f16,  register>, !fly.layout<4:1>) -> !fly.memref<f16,  register, 4:1>

    fly.copy_atom_call(%copy, %reg_view, %dst_view)
        : (!fly.copy_atom<!fly.universal_copy<64>, 16>,
           !fly.memref<f16,  register, 4:1>,
           !fly.memref<f16,  global, 4:8>) -> ()
    gpu.return
  }
}
