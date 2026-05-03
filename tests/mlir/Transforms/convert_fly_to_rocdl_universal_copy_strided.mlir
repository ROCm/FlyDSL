// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: not %fly-opt %s --fly-convert-atom-call-to-ssa-form --convert-fly-to-rocdl 2>&1 | FileCheck %s

gpu.module @bug_strided_universal_copy {

// CHECK: error: 'fly.copy_atom_call' op src memref contiguous bit count 16 is smaller than copy granularity 64
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
}
