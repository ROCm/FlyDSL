// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-rocdl | FileCheck %s
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-rocdl --convert-arith-to-llvm --canonicalize | FileCheck %s --check-prefix=LLVM

// CDNA4 async direct-to-LDS DMA atoms:
//   cdna4.buffer_load_async_lds (buffer_desc -> shared, stateful)
//   cdna4.global_load_async_lds (global -> shared, stateless)

// -----

// BufferLoadAsyncLDS state struct is {i32, i32} (soffset, imm_offset)

// CHECK-LABEL: @test_buffer_load_async_lds_type
// CHECK-SAME: (%{{.*}}: !llvm.struct<(i32, i32)>)
func.func @test_buffer_load_async_lds_type(
    %atom: !fly.copy_atom<!fly_rocdl.cdna4.buffer_load_async_lds<32>, 32>) {
  return
}

// -----

// make_copy_atom produces default state (soffset = 0, imm_offset = 0)

// CHECK-LABEL: @test_make_buffer_load_async_lds_default
// LLVM-LABEL: @test_make_buffer_load_async_lds_default
func.func @test_make_buffer_load_async_lds_default(
    %src: !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>,
    %dst: !fly.memref<f32, shared, 1:1>) {
  // CHECK-DAG: %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32)>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[S1:.*]] = llvm.insertvalue %[[C0]], %[[UNDEF]][0]
  // CHECK: %[[ATOM:.*]] = llvm.insertvalue %[[C0]], %[[S1]][1]
  %atom = fly.make_copy_atom {valBits = 32 : i32} : !fly.copy_atom<!fly_rocdl.cdna4.buffer_load_async_lds<32>, 32>
  // CHECK: rocdl.raw.ptr.buffer.load.async.lds
  //
  // The default imm_offset folds to a constant operand, so the call satisfies
  // the intrinsic's ImmArg requirement.
  // LLVM-DAG: %[[LC0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // LLVM: rocdl.raw.ptr.buffer.load.async.lds %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[LC0]], 0
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.cdna4.buffer_load_async_lds<32>, 32>, !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>, !fly.memref<f32, shared, 1:1>) -> ()
  return
}

// -----

// Run-time soffset with a constant imm_offset (128-bit).
//
// `soffset` is a genuine run-time SSA operand of the intrinsic, so a function
// argument is fine. `imm_offset` is not: it maps to the intrinsic's ImmArg
// `offset` operand, so it must reach the op as a constant or the LLVM IR
// verifier rejects the call with "immarg operand has non-immediate parameter".
// The atom is therefore built locally (make_copy_atom seeds imm_offset with a
// constant) rather than taken as a function argument, whose state struct field
// would be a run-time value.

// CHECK-LABEL: @test_buffer_load_async_lds_soffset_and_const_imm
// LLVM-LABEL: @test_buffer_load_async_lds_soffset_and_const_imm
func.func @test_buffer_load_async_lds_soffset_and_const_imm(
    %soff: i32,
    %src: !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>,
    %dst: !fly.memref<f32, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 128 : i32} : !fly.copy_atom<!fly_rocdl.cdna4.buffer_load_async_lds<128>, 128>
  %c256 = arith.constant 256 : i32
  %a1 = fly.atom.set_value(%atom, "soffset", %soff) : (!fly.copy_atom<!fly_rocdl.cdna4.buffer_load_async_lds<128>, 128>, i32) -> !fly.copy_atom<!fly_rocdl.cdna4.buffer_load_async_lds<128>, 128>
  %a2 = fly.atom.set_value(%a1, "imm_offset", %c256) : (!fly.copy_atom<!fly_rocdl.cdna4.buffer_load_async_lds<128>, 128>, i32) -> !fly.copy_atom<!fly_rocdl.cdna4.buffer_load_async_lds<128>, 128>
  // CHECK: %[[SOFF_RAW:.*]] = llvm.extractvalue %{{.*}}[0]
  // CHECK: %[[IMM_OFF:.*]] = llvm.extractvalue %{{.*}}[1]
  // CHECK: %[[SOFF_BYTES:.*]] = arith.muli %[[SOFF_RAW]], %{{.*}}
  // CHECK: rocdl.raw.ptr.buffer.load.async.lds %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[SOFF_BYTES]], %[[IMM_OFF]]
  //
  // After lowering, imm_offset really is a constant operand, not an extractvalue.
  // LLVM-DAG: %[[C256:.*]] = llvm.mlir.constant(256 : i32) : i32
  // LLVM: rocdl.raw.ptr.buffer.load.async.lds %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[C256]], 0
  fly.copy_atom_call(%a2, %src, %dst) : (!fly.copy_atom<!fly_rocdl.cdna4.buffer_load_async_lds<128>, 128>, !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>, !fly.memref<f32, shared, 1:1>) -> ()
  return
}

// -----

// Constant soffset and imm_offset both inline after lowering to LLVM

// LLVM-LABEL: @test_buffer_load_async_lds_const_both
func.func @test_buffer_load_async_lds_const_both(
    %src: !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>,
    %dst: !fly.memref<f32, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 32 : i32} : !fly.copy_atom<!fly_rocdl.cdna4.buffer_load_async_lds<32>, 32>
  %c64 = arith.constant 64 : i32
  %c256 = arith.constant 256 : i32
  %a1 = fly.atom.set_value(%atom, "soffset", %c64) : (!fly.copy_atom<!fly_rocdl.cdna4.buffer_load_async_lds<32>, 32>, i32) -> !fly.copy_atom<!fly_rocdl.cdna4.buffer_load_async_lds<32>, 32>
  %a2 = fly.atom.set_value(%a1, "imm_offset", %c256) : (!fly.copy_atom<!fly_rocdl.cdna4.buffer_load_async_lds<32>, 32>, i32) -> !fly.copy_atom<!fly_rocdl.cdna4.buffer_load_async_lds<32>, 32>
  // soffset = 64 * 4 (f32 elem bytes), imm_offset = 256 (constant); the trailing
  // `0` is the cache-policy attribute, not an SSA operand.
  // LLVM-DAG: %[[C4:.*]] = llvm.mlir.constant(4 : i32) : i32
  // LLVM-DAG: %[[C64:.*]] = llvm.mlir.constant(64 : i32) : i32
  // LLVM-DAG: %[[C256:.*]] = llvm.mlir.constant(256 : i32) : i32
  // LLVM: %[[SOFF:.*]] = llvm.mul %[[C64]], %[[C4]]
  // LLVM: rocdl.raw.ptr.buffer.load.async.lds %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[SOFF]], %[[C256]], 0
  fly.copy_atom_call(%a2, %src, %dst) : (!fly.copy_atom<!fly_rocdl.cdna4.buffer_load_async_lds<32>, 32>, !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>, !fly.memref<f32, shared, 1:1>) -> ()
  return
}

// -----

// GlobalLoadAsyncLDS is stateless: no atom state struct operand.

// CHECK-LABEL: @test_global_load_async_lds_default
func.func @test_global_load_async_lds_default(
    %src: !fly.memref<f32, global, 1:1>,
    %dst: !fly.memref<f32, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 32 : i32} : !fly.copy_atom<!fly_rocdl.cdna4.global_load_async_lds<32>, 32>
  // CHECK: rocdl.global.load.async.lds %{{.*}}, %{{.*}}, 4, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.cdna4.global_load_async_lds<32>, 32>, !fly.memref<f32, global, 1:1>, !fly.memref<f32, shared, 1:1>) -> ()
  return
}

// -----

// 128-bit global async load: size attribute is 16 bytes.

// CHECK-LABEL: @test_global_load_async_lds_128b
func.func @test_global_load_async_lds_128b(
    %src: !fly.memref<f32, global, 1:1>,
    %dst: !fly.memref<f32, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 128 : i32} : !fly.copy_atom<!fly_rocdl.cdna4.global_load_async_lds<128>, 128>
  // CHECK: rocdl.global.load.async.lds %{{.*}}, %{{.*}}, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.cdna4.global_load_async_lds<128>, 128>, !fly.memref<f32, global, 1:1>, !fly.memref<f32, shared, 1:1>) -> ()
  return
}

// -----

// 96-bit (12-byte, gfx950 DWORDX3) is a valid LDS DMA width on CDNA4.

// CHECK-LABEL: @test_global_load_async_lds_96b
func.func @test_global_load_async_lds_96b(
    %src: !fly.memref<f32, global, 1:1>,
    %dst: !fly.memref<f32, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 96 : i32} : !fly.copy_atom<!fly_rocdl.cdna4.global_load_async_lds<96>, 96>
  // CHECK: rocdl.global.load.async.lds %{{.*}}, %{{.*}}, 12, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.cdna4.global_load_async_lds<96>, 96>, !fly.memref<f32, global, 1:1>, !fly.memref<f32, shared, 1:1>) -> ()
  return
}
