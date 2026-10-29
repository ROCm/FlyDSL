// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --split-input-file --fly-canonicalize --fly-layout-lowering --verify-diagnostics

// A predicate that omits the value mode.
gpu.module @m {
  gpu.func @pred_omit_v_two_atoms(%src_ptr: !fly.ptr<f32, global>, %dst_ptr: !fly.ptr<f32, register>,
                                  %pred_ptr: !fly.ptr<i1, register>) kernel {
    %s = fly.make_int_tuple() : () -> !fly.int_tuple<((4,2),2)>
    %d = fly.make_int_tuple() : () -> !fly.int_tuple<((1,4),8)>
    %l = fly.make_layout(%s, %d) : (!fly.int_tuple<((4,2),2)>, !fly.int_tuple<((1,4),8)>) -> !fly.layout<((4,2),2):((1,4),8)>
    %sv = fly.make_view(%src_ptr, %l) : (!fly.ptr<f32, global>, !fly.layout<((4,2),2):((1,4),8)>) -> !fly.memref<f32, global, ((4,2),2):((1,4),8)>
    %dv = fly.make_view(%dst_ptr, %l) : (!fly.ptr<f32, register>, !fly.layout<((4,2),2):((1,4),8)>) -> !fly.memref<f32, register, ((4,2),2):((1,4),8)>
    %ps = fly.make_int_tuple() : () -> !fly.int_tuple<2>
    %pd = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %pl = fly.make_layout(%ps, %pd) : (!fly.int_tuple<2>, !fly.int_tuple<1>) -> !fly.layout<2:1>
    %pv = fly.make_view(%pred_ptr, %pl) : (!fly.ptr<i1, register>, !fly.layout<2:1>) -> !fly.memref<i1, register, 2:1>
    %atom = fly.make_copy_atom {valBits = 32 : i32} : !fly.copy_atom<!fly.universal_copy<128>, 32>
    // expected-error @+2 {{copy predicate of shape 2 needs the source shape ((4,2),2) or one predicate per atom ((1,2),2)}}
    // expected-error @+1 {{failed to legalize operation 'fly.copy'}}
    fly.copy(%atom, %sv, %dv, %pv)
        : (!fly.copy_atom<!fly.universal_copy<128>, 32>,
           !fly.memref<f32, global, ((4,2),2):((1,4),8)>,
           !fly.memref<f32, register, ((4,2),2):((1,4),8)>,
           !fly.memref<i1, register, 2:1>) -> ()
    gpu.return
  }
}

// -----

// A predicate whose value mode has size one.
gpu.module @m {
  gpu.func @pred_unit_v_two_atoms(%src_ptr: !fly.ptr<f32, global>, %dst_ptr: !fly.ptr<f32, register>,
                                  %pred_ptr: !fly.ptr<i1, register>) kernel {
    %s = fly.make_int_tuple() : () -> !fly.int_tuple<((4,2),2)>
    %d = fly.make_int_tuple() : () -> !fly.int_tuple<((1,4),8)>
    %l = fly.make_layout(%s, %d) : (!fly.int_tuple<((4,2),2)>, !fly.int_tuple<((1,4),8)>) -> !fly.layout<((4,2),2):((1,4),8)>
    %sv = fly.make_view(%src_ptr, %l) : (!fly.ptr<f32, global>, !fly.layout<((4,2),2):((1,4),8)>) -> !fly.memref<f32, global, ((4,2),2):((1,4),8)>
    %dv = fly.make_view(%dst_ptr, %l) : (!fly.ptr<f32, register>, !fly.layout<((4,2),2):((1,4),8)>) -> !fly.memref<f32, register, ((4,2),2):((1,4),8)>
    %ps = fly.make_int_tuple() : () -> !fly.int_tuple<(1,2)>
    %pd = fly.make_int_tuple() : () -> !fly.int_tuple<(0,1)>
    %pl = fly.make_layout(%ps, %pd) : (!fly.int_tuple<(1,2)>, !fly.int_tuple<(0,1)>) -> !fly.layout<(1,2):(0,1)>
    %pv = fly.make_view(%pred_ptr, %pl) : (!fly.ptr<i1, register>, !fly.layout<(1,2):(0,1)>) -> !fly.memref<i1, register, (1,2):(0,1)>
    %atom = fly.make_copy_atom {valBits = 32 : i32} : !fly.copy_atom<!fly.universal_copy<128>, 32>
    // expected-error @+2 {{copy predicate of shape (1,2) needs the source shape ((4,2),2) or one predicate per atom ((1,2),2)}}
    // expected-error @+1 {{failed to legalize operation 'fly.copy'}}
    fly.copy(%atom, %sv, %dv, %pv)
        : (!fly.copy_atom<!fly.universal_copy<128>, 32>,
           !fly.memref<f32, global, ((4,2),2):((1,4),8)>,
           !fly.memref<f32, register, ((4,2),2):((1,4),8)>,
           !fly.memref<i1, register, (1,2):(0,1)>) -> ()
    gpu.return
  }
}
