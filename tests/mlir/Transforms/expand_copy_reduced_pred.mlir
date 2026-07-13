// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-layout-lowering | FileCheck %s

// CHECK-LABEL: gpu.func @reduced_pred_copy
// CHECK: fly.copy_atom_call
// CHECK-SAME: !fly.memref<f32, #fly_rocdl.buffer_desc, 4:1>, !fly.memref<f32, register, 4:1>, !fly.memref<i1, register, 1:0>) -> ()
gpu.module @expand_copy_reduced_pred {
  gpu.func @reduced_pred_copy(%src: !fly.ptr<f32, global>, %dst_ptr: !fly.ptr<f32, register>,
                              %pred_ptr: !fly.ptr<i1, register>) kernel {
    %c0_i16 = arith.constant 0 : i16
    %c4294967295_i64 = arith.constant 4294967295 : i64
    %c159744_i32 = arith.constant 159744 : i32

    // src: buffer-descriptor view with V-mode (4,4) = (ATOM_V, ATOM_REST) and unit REST.
    %src_shape = fly.make_int_tuple() : () -> !fly.int_tuple<((4,4),1,1)>
    %src_stride = fly.make_int_tuple() : () -> !fly.int_tuple<((1,200),0,0)>
    %src_layout = fly.make_layout(%src_shape, %src_stride)
        : (!fly.int_tuple<((4,4),1,1)>, !fly.int_tuple<((1,200),0,0)>)
        -> !fly.layout<((4,4),1,1):((1,200),0,0)>
    %src_desc = fly.make_ptr(%src, %c0_i16, %c4294967295_i64, %c159744_i32)
        : (!fly.ptr<f32, global>, i16, i64, i32) -> !fly.ptr<f32, #fly_rocdl.buffer_desc>
    %src_view = fly.make_view(%src_desc, %src_layout)
        : (!fly.ptr<f32, #fly_rocdl.buffer_desc>, !fly.layout<((4,4),1,1):((1,200),0,0)>)
        -> !fly.memref<f32, #fly_rocdl.buffer_desc, ((4,4),1,1):((1,200),0,0)>

    // dst: register fragment, same V-mode (4,4).
    %dst_shape = fly.make_int_tuple() : () -> !fly.int_tuple<((4,4),1,1)>
    %dst_stride = fly.make_int_tuple() : () -> !fly.int_tuple<((1,4),0,0)>
    %dst_layout = fly.make_layout(%dst_shape, %dst_stride)
        : (!fly.int_tuple<((4,4),1,1)>, !fly.int_tuple<((1,4),0,0)>)
        -> !fly.layout<((4,4),1,1):((1,4),0,0)>
    %dst_view = fly.make_view(%dst_ptr, %dst_layout)
        : (!fly.ptr<f32, register>, !fly.layout<((4,4),1,1):((1,4),0,0)>)
        -> !fly.memref<f32, register, ((4,4),1,1):((1,4),0,0)>

    // pred: reduced (ATOM_REST, REST...) = (4,1,1), one boolean per atom (ATOM_V dropped).
    %pred_shape = fly.make_int_tuple() : () -> !fly.int_tuple<(4,1,1)>
    %pred_stride = fly.make_int_tuple() : () -> !fly.int_tuple<(1,0,0)>
    %pred_layout = fly.make_layout(%pred_shape, %pred_stride)
        : (!fly.int_tuple<(4,1,1)>, !fly.int_tuple<(1,0,0)>) -> !fly.layout<(4,1,1):(1,0,0)>
    %pred_view = fly.make_view(%pred_ptr, %pred_layout)
        : (!fly.ptr<i1, register>, !fly.layout<(4,1,1):(1,0,0)>)
        -> !fly.memref<i1, register, (4,1,1):(1,0,0)>

    %atom = fly.make_copy_atom {valBits = 32 : i32} : !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<128>, 32>
    fly.copy(%atom, %src_view, %dst_view, %pred_view)
        : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<128>, 32>,
           !fly.memref<f32, #fly_rocdl.buffer_desc, ((4,4),1,1):((1,200),0,0)>,
           !fly.memref<f32, register, ((4,4),1,1):((1,4),0,0)>,
           !fly.memref<i1, register, (4,1,1):(1,0,0)>) -> ()
    gpu.return
  }
}
