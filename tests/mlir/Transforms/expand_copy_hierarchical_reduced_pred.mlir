// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-layout-lowering | FileCheck %s --check-prefix=LAYOUT
// RUN: %fly-opt %s --fly-layout-lowering --fly-convert-atom-call-to-ssa-form --fly-promote-regmem-to-vectorssa | FileCheck %s --check-prefix=SSA

// Regression test for a reduced predicate whose ATOM_REST is hierarchical.
//
// Source V-mode: (ATOM_V, ATOM_REST) = (1, (2, 2))
// Predicate:     ATOM_REST           =     (2, 2)
//
// Rank-based reduced-predicate detection confused the two rank-2 shapes and
// generated vector.extract_strided_slice offset=4,size=2 from vector<4xi1>.

// LAYOUT-LABEL: gpu.func @hierarchical_reduced_pred_copy
// LAYOUT-COUNT-4: fly.copy_atom_call
// LAYOUT-NOT: fly.copy(
// SSA-LABEL: gpu.func @hierarchical_reduced_pred_copy
// SSA-COUNT-4: fly.copy_atom_call_ssa
// SSA-NOT: vector.extract_strided_slice
gpu.module @expand_copy_hierarchical_reduced_pred {
  gpu.func @hierarchical_reduced_pred_copy(
      %src_ptr: !fly.ptr<f32, global>,
      %dst_ptr: !fly.ptr<f32, register>,
      %pred_ptr: !fly.ptr<i1, register>) kernel {
    %src_shape = fly.make_int_tuple()
        : () -> !fly.int_tuple<((1,(2,2)),1,1)>
    %src_stride = fly.make_int_tuple()
        : () -> !fly.int_tuple<((0,(1,2)),0,0)>
    %src_layout = fly.make_layout(%src_shape, %src_stride)
        : (!fly.int_tuple<((1,(2,2)),1,1)>, !fly.int_tuple<((0,(1,2)),0,0)>)
        -> !fly.layout<((1,(2,2)),1,1):((0,(1,2)),0,0)>
    %src = fly.make_view(%src_ptr, %src_layout)
        : (!fly.ptr<f32, global>, !fly.layout<((1,(2,2)),1,1):((0,(1,2)),0,0)>)
        -> !fly.memref<f32, global, ((1,(2,2)),1,1):((0,(1,2)),0,0)>

    %dst_layout = fly.make_layout(%src_shape, %src_stride)
        : (!fly.int_tuple<((1,(2,2)),1,1)>, !fly.int_tuple<((0,(1,2)),0,0)>)
        -> !fly.layout<((1,(2,2)),1,1):((0,(1,2)),0,0)>
    %dst = fly.make_view(%dst_ptr, %dst_layout)
        : (!fly.ptr<f32, register>, !fly.layout<((1,(2,2)),1,1):((0,(1,2)),0,0)>)
        -> !fly.memref<f32, register, ((1,(2,2)),1,1):((0,(1,2)),0,0)>

    %pred_shape = fly.make_int_tuple()
        : () -> !fly.int_tuple<((2,2),1,1)>
    %pred_stride = fly.make_int_tuple()
        : () -> !fly.int_tuple<((1,2),0,0)>
    %pred_layout = fly.make_layout(%pred_shape, %pred_stride)
        : (!fly.int_tuple<((2,2),1,1)>, !fly.int_tuple<((1,2),0,0)>)
        -> !fly.layout<((2,2),1,1):((1,2),0,0)>
    %pred = fly.make_view(%pred_ptr, %pred_layout)
        : (!fly.ptr<i1, register>, !fly.layout<((2,2),1,1):((1,2),0,0)>)
        -> !fly.memref<i1, register, ((2,2),1,1):((1,2),0,0)>

    %atom = fly.make_copy_atom {valBits = 32 : i32}
        : !fly.copy_atom<!fly.universal_copy<32>, 32>
    fly.copy(%atom, %src, %dst, %pred)
        : (!fly.copy_atom<!fly.universal_copy<32>, 32>,
           !fly.memref<f32, global, ((1,(2,2)),1,1):((0,(1,2)),0,0)>,
           !fly.memref<f32, register, ((1,(2,2)),1,1):((0,(1,2)),0,0)>,
           !fly.memref<i1, register, ((2,2),1,1):((1,2),0,0)>) -> ()
    gpu.return
  }
}
