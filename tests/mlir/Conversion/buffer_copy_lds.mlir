// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-rocdl | FileCheck %s
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-rocdl --convert-arith-to-llvm --canonicalize | FileCheck %s --check-prefix=LLVM

// BufferCopyLDS (buffer_desc -> shared) lowering tests

// -----

// State struct is {i32, i32} (soffset, imm_offset)

// CHECK-LABEL: @test_buffer_copy_lds_type
// CHECK-SAME: (%{{.*}}: !llvm.struct<(i32, i32)>)
func.func @test_buffer_copy_lds_type(
    %atom: !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<32>, 32>) {
  return
}

// -----

// make_copy_atom produces default state (soffset = 0, imm_offset = 0)

// CHECK-LABEL: @test_make_copy_atom_lds_default
func.func @test_make_copy_atom_lds_default(
    %src: !fly.memref<f32, buffer_desc, 1:1>,
    %dst: !fly.memref<f32, shared, 1:1>) {
  // CHECK-DAG: %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32)>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[S1:.*]] = llvm.insertvalue %[[C0]], %[[UNDEF]][0]
  // CHECK: %[[ATOM:.*]] = llvm.insertvalue %[[C0]], %[[S1]][1]
  %atom = fly.make_copy_atom {valBits = 32 : i32} : !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<32>, 32>
  // CHECK: rocdl.raw.ptr.buffer.load.lds
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<32>, 32>, !fly.memref<f32, buffer_desc, 1:1>, !fly.memref<f32, shared, 1:1>) -> ()
  return
}

// -----

// set soffset then copy: soffset flows into rocdl.raw.ptr.buffer.load.lds

// CHECK-LABEL: @test_buffer_copy_lds_set_soffset
// CHECK-SAME: (%[[ATOM:.*]]: !llvm.struct<(i32, i32)>, %[[SOFF:.*]]: i32,
func.func @test_buffer_copy_lds_set_soffset(
    %atom: !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<32>, 32>,
    %soff: i32,
    %src: !fly.memref<f32, buffer_desc, 1:1>,
    %dst: !fly.memref<f32, shared, 1:1>) {
  // CHECK: %[[NEW_ATOM:.*]] = llvm.insertvalue %[[SOFF]], %[[ATOM]][0]
  %new_atom = fly.atom.set_value(%atom, "soffset", %soff) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<32>, 32>, i32) -> !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<32>, 32>
  // CHECK: %[[SOFF_RAW:.*]] = llvm.extractvalue %[[NEW_ATOM]][0]
  // CHECK: %[[IMM_OFF:.*]] = llvm.extractvalue %[[NEW_ATOM]][1]
  // CHECK: %[[SOFF_BYTES:.*]] = arith.muli %[[SOFF_RAW]], %{{.*}}
  // CHECK: rocdl.raw.ptr.buffer.load.lds %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[SOFF_BYTES]], %[[IMM_OFF]]
  fly.copy_atom_call(%new_atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<32>, 32>, !fly.memref<f32, buffer_desc, 1:1>, !fly.memref<f32, shared, 1:1>) -> ()
  return
}

// -----

// set imm_offset then copy: imm_offset flows into rocdl.raw.ptr.buffer.load.lds offset arg

// CHECK-LABEL: @test_buffer_copy_lds_set_imm_offset
// CHECK-SAME: (%[[ATOM:.*]]: !llvm.struct<(i32, i32)>, %[[IMM:.*]]: i32,
func.func @test_buffer_copy_lds_set_imm_offset(
    %atom: !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<32>, 32>,
    %imm: i32,
    %src: !fly.memref<f32, buffer_desc, 1:1>,
    %dst: !fly.memref<f32, shared, 1:1>) {
  // CHECK: %[[NEW_ATOM:.*]] = llvm.insertvalue %[[IMM]], %[[ATOM]][1]
  %new_atom = fly.atom.set_value(%atom, "imm_offset", %imm) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<32>, 32>, i32) -> !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<32>, 32>
  // CHECK: %[[SOFF_RAW:.*]] = llvm.extractvalue %[[NEW_ATOM]][0]
  // CHECK: %[[IMM_OFF:.*]] = llvm.extractvalue %[[NEW_ATOM]][1]
  // CHECK: %[[SOFF_BYTES:.*]] = arith.muli %[[SOFF_RAW]], %{{.*}}
  // CHECK: rocdl.raw.ptr.buffer.load.lds %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[SOFF_BYTES]], %[[IMM_OFF]]
  fly.copy_atom_call(%new_atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<32>, 32>, !fly.memref<f32, buffer_desc, 1:1>, !fly.memref<f32, shared, 1:1>) -> ()
  return
}

// -----

// set both soffset and imm_offset then copy (128-bit)

// CHECK-LABEL: @test_buffer_copy_lds_set_both
func.func @test_buffer_copy_lds_set_both(
    %atom: !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<128>, 128>,
    %soff: i32,
    %imm: i32,
    %src: !fly.memref<f32, buffer_desc, 1:1>,
    %dst: !fly.memref<f32, shared, 1:1>) {
  %a1 = fly.atom.set_value(%atom, "soffset", %soff) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<128>, 128>, i32) -> !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<128>, 128>
  %a2 = fly.atom.set_value(%a1, "imm_offset", %imm) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<128>, 128>, i32) -> !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<128>, 128>
  // CHECK: %[[SOFF_RAW:.*]] = llvm.extractvalue %{{.*}}[0]
  // CHECK: %[[IMM_OFF:.*]] = llvm.extractvalue %{{.*}}[1]
  // CHECK: %[[SOFF_BYTES:.*]] = arith.muli %[[SOFF_RAW]], %{{.*}}
  // CHECK: rocdl.raw.ptr.buffer.load.lds %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[SOFF_BYTES]], %[[IMM_OFF]]
  fly.copy_atom_call(%a2, %src, %dst) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<128>, 128>, !fly.memref<f32, buffer_desc, 1:1>, !fly.memref<f32, shared, 1:1>) -> ()
  return
}

// -----

// Constant imm_offset inlines as an immediate after lowering to LLVM

// LLVM-LABEL: @test_buffer_copy_lds_const_imm_offset
func.func @test_buffer_copy_lds_const_imm_offset(
    %src: !fly.memref<f32, buffer_desc, 1:1>,
    %dst: !fly.memref<f32, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 32 : i32} : !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<32>, 32>
  %c128 = arith.constant 128 : i32
  %a1 = fly.atom.set_value(%atom, "imm_offset", %c128) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<32>, 32>, i32) -> !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<32>, 32>
  // LLVM-DAG: %[[C128:.*]] = llvm.mlir.constant(128 : i32) : i32
  // LLVM-DAG: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // LLVM: rocdl.raw.ptr.buffer.load.lds %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[C128]], %[[C0]]
  fly.copy_atom_call(%a1, %src, %dst) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<32>, 32>, !fly.memref<f32, buffer_desc, 1:1>, !fly.memref<f32, shared, 1:1>) -> ()
  return
}

// -----

// Constant soffset and imm_offset both inline after lowering

// LLVM-LABEL: @test_buffer_copy_lds_const_both
func.func @test_buffer_copy_lds_const_both(
    %src: !fly.memref<f32, buffer_desc, 1:1>,
    %dst: !fly.memref<f32, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 32 : i32} : !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<32>, 32>
  %c64 = arith.constant 64 : i32
  %c256 = arith.constant 256 : i32
  %a1 = fly.atom.set_value(%atom, "soffset", %c64) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<32>, 32>, i32) -> !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<32>, 32>
  %a2 = fly.atom.set_value(%a1, "imm_offset", %c256) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<32>, 32>, i32) -> !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<32>, 32>
  // soffset = 64 * 4 (f32 elem bytes), imm_offset = 256 (constant)
  // LLVM-DAG: %[[C4:.*]] = llvm.mlir.constant(4 : i32) : i32
  // LLVM-DAG: %[[C64:.*]] = llvm.mlir.constant(64 : i32) : i32
  // LLVM-DAG: %[[C256:.*]] = llvm.mlir.constant(256 : i32) : i32
  // LLVM-DAG: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // LLVM: %[[SOFF:.*]] = llvm.mul %[[C64]], %[[C4]]
  // LLVM: rocdl.raw.ptr.buffer.load.lds %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[SOFF]], %[[C256]], %[[C0]]
  fly.copy_atom_call(%a2, %src, %dst) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<32>, 32>, !fly.memref<f32, buffer_desc, 1:1>, !fly.memref<f32, shared, 1:1>) -> ()
  return
}
