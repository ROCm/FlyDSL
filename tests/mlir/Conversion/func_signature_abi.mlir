// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --split-input-file --fly-layout-lowering --convert-fly-to-rocdl | FileCheck %s

// Fly values cross function, call, return and launch boundaries as packed structs of their
// dynamic leaves; static values and static fields are dropped. These are the cases of the
// former fly-rewrite-func-signature test, now converted by convert-fly-to-rocdl.

// CHECK-LABEL: func.func @test_sink_static_int_tuple(
// CHECK-SAME: )
// CHECK: return
// Each single-arg test body holds a small consumer (fly.get_* / fly.get_iter)
// that uses the DSL argument: this is what triggers the source materialization
// (fly.static / llvm.extractvalue / fly.make_*) inserted by the DialectConversion
// framework at the function entry. Without a real use, no materialization is
// emitted.

// === Sink static DSL args from func ===

// Static int_tuple arg is removed; fly.static materializes it at the use.
func.func @test_sink_static_int_tuple(%arg0: !fly.int_tuple<(4,8)>) {
  %0 = fly.get(%arg0) {mode = array<i32: 0>} : (!fly.int_tuple<(4,8)>) -> !fly.int_tuple<4>
  return
}

// -----

// CHECK-LABEL: func.func @test_sink_static_layout(
// CHECK-SAME: )
// CHECK: return
// Static layout arg is removed.
func.func @test_sink_static_layout(%arg0: !fly.layout<(4,8):(1,4)>) {
  %0 = fly.get_shape(%arg0) : (!fly.layout<(4,8):(1,4)>) -> !fly.int_tuple<(4,8)>
  return
}

// -----

// CHECK-LABEL: func.func @test_sink_mixed(
// CHECK-SAME: %{{.*}}: i32)
// CHECK: return
// Mixed: static DSL arg removed, non-DSL arg kept.
func.func @test_sink_mixed(%arg0: !fly.int_tuple<5>, %arg1: i32) {
  %0 = fly.get_scalar(%arg0) : (!fly.int_tuple<5>) -> i32
  return
}

// -----

// CHECK-LABEL: func.func @test_dynamic_int_tuple(
// CHECK-SAME: %{{.*}}: !llvm.struct<packed (i32)>)
// CHECK: return
// === Rewrite dynamic DSL args to LLVM struct ===

// Dynamic int_tuple arg becomes packed struct of its dynamic leaves.
func.func @test_dynamic_int_tuple(%arg0: !fly.int_tuple<?>) {
  %0 = fly.get_scalar(%arg0) : (!fly.int_tuple<?>) -> i32
  return
}

// -----

// CHECK-LABEL: func.func @test_dynamic_nested_int_tuple(
// CHECK-SAME: %{{.*}}: !llvm.struct<packed (i32, i32)>)
// CHECK: return
// Nested dynamic int_tuple: two dynamic leaves -> struct with two i32 fields.
func.func @test_dynamic_nested_int_tuple(%arg0: !fly.int_tuple<(?,?)>) {
  %0 = fly.get(%arg0) {mode = array<i32: 0>} : (!fly.int_tuple<(?,?)>) -> !fly.int_tuple<?>
  return
}

// -----

// CHECK-LABEL: func.func @test_dynamic_layout(
// CHECK-SAME: %{{.*}}: !llvm.struct<packed (struct<packed (i32)>, struct<packed (i32)>)>)
// CHECK: return
// Layout with dynamic shape and stride: struct contains two sub-structs.
func.func @test_dynamic_layout(%arg0: !fly.layout<?:?>) {
  %0 = fly.get_shape(%arg0) : (!fly.layout<?:?>) -> !fly.int_tuple<?>
  return
}

// -----

// CHECK-LABEL: func.func @test_partially_dynamic_layout(
// CHECK-SAME: %{{.*}}: !llvm.struct<packed (struct<packed (i32)>)>)
// CHECK: return
// Layout with only dynamic stride; shape is static -> struct has one sub-struct.
func.func @test_partially_dynamic_layout(%arg0: !fly.layout<4:?>) {
  %0 = fly.get_shape(%arg0) : (!fly.layout<4:?>) -> !fly.int_tuple<4>
  return
}

// -----

// CHECK-LABEL: func.func @test_static_memref(
// CHECK-SAME: %{{.*}}: !llvm.ptr<1>)
// CHECK: return
// MemRef with static layout: lowered to a single fly.ptr argument.
func.func @test_static_memref(%arg0: !fly.memref<f32, global, 32:1>) {
  %0 = fly.get_iter(%arg0) : (!fly.memref<f32, global, 32:1>) -> !fly.ptr<f32, global>
  return
}

// -----

// CHECK-LABEL: func.func @test_dynamic_memref(
// CHECK-SAME: %{{.*}}: !llvm.ptr<3>, %{{.*}}: !llvm.struct<packed (struct<packed (i32)>, struct<packed (i32)>)>)
// CHECK: return
// MemRef with dynamic layout: lowered to ptr arg + layout struct arg.
func.func @test_dynamic_memref(%arg0: !fly.memref<f16, shared, (16,?):(1,?)>) {
  %0 = fly.get_iter(%arg0) : (!fly.memref<f16, shared, (16,?):(1,?)>) -> !fly.ptr<f16, shared>
  return
}

// -----

// CHECK-LABEL: func.func @test_passthrough(
// CHECK-SAME: %{{.*}}: i32, %{{.*}}: f32)
// CHECK: return
// Non-DSL args are passed through unchanged.
func.func @test_passthrough(%arg0: i32, %arg1: f32) {
  return
}

// -----

// CHECK-LABEL: func.func private @test_declaration_dynamic(
// CHECK-SAME: !llvm.struct<packed (i32, i32)>)
// Declaration without body: signature is rewritten but no unpack code is generated.
func.func private @test_declaration_dynamic(!fly.int_tuple<(?,?)>)

// -----

// CHECK-LABEL: func.func private @test_declaration_static(
// CHECK-SAME: )
// Declaration with static arg: arg is removed entirely.
func.func private @test_declaration_static(!fly.int_tuple<(4,8)>)

// -----

// CHECK-LABEL: func.func @test_return_dynamic_memref(
// CHECK-SAME: %{{.*}}: !llvm.ptr<3>, %{{.*}}: i32) -> (!llvm.ptr<3>, !llvm.struct<packed (struct<packed (i32)>, struct<packed (i32)>)>)
// CHECK: return %{{.*}}, %{{.*}} : !llvm.ptr<3>, !llvm.struct<packed (struct<packed (i32)>, struct<packed (i32)>)>
// === Function results (regression coverage) ===
//
// Result types are converted together with arg types, and `func.return` is
// rewritten in lock-step by populateReturnOpTypeConversionPattern. Real
// kernels usually return void, but these cases make sure the pass does not
// silently rely on that invariant: if the pattern is ever dropped, the
// verifier failure on the rewritten return will land here.

// Returning a dynamic-layout memref: result becomes a (ptr, layout struct)
// pair and the return is rewritten to yield both carrier values.
func.func @test_return_dynamic_memref(%p: !fly.ptr<f16, shared>, %m: i32) -> !fly.memref<f16, shared, (16,?):(1,?)> {
  %sh = fly.make_int_tuple(%m) {static_args = array<i64: 16, -1>} : (i32) -> !fly.int_tuple<(16,?)>
  %sn = fly.make_int_tuple(%m) {static_args = array<i64: 1, -1>} : (i32) -> !fly.int_tuple<(1,?)>
  %l = fly.make_layout(%sh, %sn) : (!fly.int_tuple<(16,?)>, !fly.int_tuple<(1,?)>) -> !fly.layout<(16,?):(1,?)>
  %v = fly.make_view(%p, %l) : (!fly.ptr<f16, shared>, !fly.layout<(16,?):(1,?)>) -> !fly.memref<f16, shared, (16,?):(1,?)>
  return %v : !fly.memref<f16, shared, (16,?):(1,?)>
}

// -----

// CHECK-LABEL: func.func @test_return_static_memref(
// CHECK-SAME: %{{.*}}: !llvm.ptr<1>) -> !llvm.ptr<1>
// CHECK: return %{{.*}} : !llvm.ptr<1>
// Returning a static-layout memref: result is a single ptr carrier.
func.func @test_return_static_memref(%p: !fly.ptr<f32, global>) -> !fly.memref<f32, global, 32:1> {
  %l = fly.static : !fly.layout<32:1>
  %v = fly.make_view(%p, %l) : (!fly.ptr<f32, global>, !fly.layout<32:1>) -> !fly.memref<f32, global, 32:1>
  return %v : !fly.memref<f32, global, 32:1>
}

// -----

// CHECK-LABEL: func.func @test_return_static_int_tuple(
// CHECK-SAME: )
// CHECK: return
// Returning a fully-static DSL value: result disappears entirely (1:0) and
// the return op loses its operand.
func.func @test_return_static_int_tuple() -> !fly.int_tuple<(4,8)> {
  %x = fly.static : !fly.int_tuple<(4,8)>
  return %x : !fly.int_tuple<(4,8)>
}

// -----

// CHECK-LABEL: func.func private @callee_dynamic_arg(
// CHECK-SAME: !llvm.ptr<3>, !llvm.struct<packed (struct<packed (i32)>, struct<packed (i32)>)>)
// CHECK-LABEL: func.func @test_call_dynamic_memref_arg(
// CHECK-SAME: %{{.*}}: !llvm.ptr<3>, %{{.*}}: i32)
// CHECK: call @callee_dynamic_arg(%{{.*}}, %{{.*}}) : (!llvm.ptr<3>, !llvm.struct<packed (struct<packed (i32)>, struct<packed (i32)>)>) -> ()
// CHECK: return
// === Call sites (regression coverage) ===
//
// When a callee's signature is rewritten, every `func.call` to it must be
// rewritten in lock-step so operand/result types keep matching the new
// signature. populateCallOpTypeConversionPattern handles the call site;
// these tests pin the behaviour for the three cardinality cases (1:N, 1:0,
// and N:1 on the result side).

// Caller passing a dynamic-layout memref: call operand gets 1:N expanded
// and the callee declaration's signature is rewritten to match.
func.func private @callee_dynamic_arg(!fly.memref<f16, shared, (16,?):(1,?)>)
func.func @test_call_dynamic_memref_arg(%p: !fly.ptr<f16, shared>, %m: i32) {
  %sh = fly.make_int_tuple(%m) {static_args = array<i64: 16, -1>} : (i32) -> !fly.int_tuple<(16,?)>
  %sn = fly.make_int_tuple(%m) {static_args = array<i64: 1, -1>} : (i32) -> !fly.int_tuple<(1,?)>
  %l = fly.make_layout(%sh, %sn) : (!fly.int_tuple<(16,?)>, !fly.int_tuple<(1,?)>) -> !fly.layout<(16,?):(1,?)>
  %v = fly.make_view(%p, %l) : (!fly.ptr<f16, shared>, !fly.layout<(16,?):(1,?)>) -> !fly.memref<f16, shared, (16,?):(1,?)>
  func.call @callee_dynamic_arg(%v) : (!fly.memref<f16, shared, (16,?):(1,?)>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func private @callee_dynamic_ret(
// CHECK-SAME: ) -> (!llvm.ptr<3>, !llvm.struct<packed (struct<packed (i32)>, struct<packed (i32)>)>)
// CHECK-LABEL: func.func @test_call_dynamic_memref_result(
// CHECK-SAME: )
// CHECK: %{{.*}}:2 = call @callee_dynamic_ret() : () -> (!llvm.ptr<3>, !llvm.struct<packed (struct<packed (i32)>, struct<packed (i32)>)>)
// CHECK: return
// Callee returns a dynamic-layout memref: call result is 1:N expanded, then
// source-materialized back to a memref for the original use site.
func.func private @callee_dynamic_ret() -> !fly.memref<f16, shared, (16,?):(1,?)>
func.func @test_call_dynamic_memref_result() {
  %v = func.call @callee_dynamic_ret() : () -> !fly.memref<f16, shared, (16,?):(1,?)>
  %p = fly.get_iter(%v) : (!fly.memref<f16, shared, (16,?):(1,?)>) -> !fly.ptr<f16, shared>
  return
}

// -----

// CHECK-LABEL: func.func private @callee_static_arg(
// CHECK-SAME: )
// CHECK-LABEL: func.func @test_call_static_arg_dropped(
// CHECK-SAME: )
// CHECK: call @callee_static_arg() : () -> ()
// CHECK: return
// Callee takes a static DSL arg (1:0): both the signature loses the param
// and the call drops the operand.
func.func private @callee_static_arg(!fly.int_tuple<(4,8)>)
func.func @test_call_static_arg_dropped() {
  %s = fly.static : !fly.int_tuple<(4,8)>
  func.call @callee_static_arg(%s) : (!fly.int_tuple<(4,8)>) -> ()
  return
}

// -----

// CHECK-LABEL: gpu.func @static_kernel(
// CHECK-SAME: ) kernel
// CHECK: gpu.return
// CHECK-LABEL: func.func @test_launch_static(
// CHECK-SAME: )
// CHECK: gpu.launch_func  @kernel_mod_static::@static_kernel blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}})
// CHECK: return
// === gpu.launch_func: pack DSL operands + drop static operands ===

module attributes {gpu.container_module} {

gpu.module @kernel_mod_static {
  gpu.func @static_kernel(%arg0: !fly.int_tuple<(4,8)>) kernel {
    // The arg is unused, so no source materialization is emitted in the body
    // (the framework only materializes when there are real uses).
    gpu.return
  }
}

// Static operands are dropped from gpu.launch_func.
func.func @test_launch_static(%t: !fly.int_tuple<(4,8)>) {
  %c1 = arith.constant 1 : index
  gpu.launch_func @kernel_mod_static::@static_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%t : !fly.int_tuple<(4,8)>)
  return
}

}

// -----

// CHECK-LABEL: gpu.func @dynamic_kernel(
// CHECK-SAME: %{{.*}}: !llvm.struct<packed (i32, i32)>) kernel
// CHECK: gpu.return
// CHECK-LABEL: func.func @test_launch_dynamic(
// CHECK-SAME: %{{.*}}: i32, %{{.*}}: i32)
// CHECK: gpu.launch_func  @kernel_mod_dynamic::@dynamic_kernel blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}})  args(%{{.*}} : !llvm.struct<packed (i32, i32)>)
// CHECK: return
module attributes {gpu.container_module} {

gpu.module @kernel_mod_dynamic {
  gpu.func @dynamic_kernel(%arg0: !fly.int_tuple<(?,?)>) kernel {
    // fly.get forces a use of %arg0 as the original DSL type, so source
    // materialization (extractvalue + make_int_tuple) fires.
    %dummy = fly.get(%arg0) {mode = array<i32: 0>} : (!fly.int_tuple<(?,?)>) -> !fly.int_tuple<?>
    gpu.return
  }
}

// Dynamic operands are packed into structs at the launch_func call site.
func.func @test_launch_dynamic(%a: i32, %b: i32) {
  %t = fly.make_int_tuple(%a, %b) : (i32, i32) -> !fly.int_tuple<(?,?)>
  %c1 = arith.constant 1 : index
  gpu.launch_func @kernel_mod_dynamic::@dynamic_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%t : !fly.int_tuple<(?,?)>)
  return
}

}

// -----

// CHECK-LABEL: func.func @test_scf_if_memref_static(
// CHECK-SAME: %{{.*}}: i1, %{{.*}}: !llvm.ptr<3>, %{{.*}}: !llvm.ptr<3>)
// CHECK: return
// === SCF control flow: expand DSL types into carriers ===

// scf.if yielding a memref (static layout) becomes scf.if yielding a fly.ptr,
// with a fresh MakeView rebuilt on the outside so downstream patterns keep
// seeing the normal form (MakeView-defined memref). Pack ops (fly.get_iter)
// are emitted at the def site of the yielded memref, not inside each branch.
// The trailing fly.get_iter sinks the scf result; it consumes %r as a memref
// so the framework inserts the outer make_view source materialization.
func.func @test_scf_if_memref_static(%c: i1, %p0: !fly.ptr<f32, shared>, %p1: !fly.ptr<f32, shared>) {
  %l = fly.static : !fly.layout<4:1>
  %v0 = fly.make_view(%p0, %l) : (!fly.ptr<f32, shared>, !fly.layout<4:1>) -> !fly.memref<f32, shared, 4:1>
  %v1 = fly.make_view(%p1, %l) : (!fly.ptr<f32, shared>, !fly.layout<4:1>) -> !fly.memref<f32, shared, 4:1>
  %r = scf.if %c -> (!fly.memref<f32, shared, 4:1>) {
    scf.yield %v0 : !fly.memref<f32, shared, 4:1>
  } else {
    scf.yield %v1 : !fly.memref<f32, shared, 4:1>
  }
  %sink = fly.get_iter(%r) : (!fly.memref<f32, shared, 4:1>) -> !fly.ptr<f32, shared>
  return
}

// -----

// CHECK-LABEL: func.func @test_scf_for_memref_static(
// CHECK-SAME: %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: !llvm.ptr<3>)
// CHECK: return
// scf.for carrying a memref through iter_args: the loop carries the fly.ptr
// carrier. When the body merely forwards the iter_arg through scf.yield (as
// below) no body-level unpack/repack is needed.
func.func @test_scf_for_memref_static(%lb: index, %ub: index, %st: index, %p: !fly.ptr<f32, shared>) {
  %l = fly.static : !fly.layout<4:1>
  %v0 = fly.make_view(%p, %l) : (!fly.ptr<f32, shared>, !fly.layout<4:1>) -> !fly.memref<f32, shared, 4:1>
  %r = scf.for %i = %lb to %ub step %st iter_args(%a = %v0) -> (!fly.memref<f32, shared, 4:1>) {
    scf.yield %a : !fly.memref<f32, shared, 4:1>
  }
  %sink = fly.get_iter(%r) : (!fly.memref<f32, shared, 4:1>) -> !fly.ptr<f32, shared>
  return
}

// -----

// CHECK-LABEL: func.func @test_scf_while_memref_static(
// CHECK-SAME: %{{.*}}: i1, %{{.*}}: !llvm.ptr<3>)
// CHECK: return
// scf.while carrying a memref: the loop carries the fly.ptr carrier, and a
// fresh MakeView is rebuilt outside so the surrounding code can still consume
// the result as a memref. As in the scf.for case, when the body merely
// forwards the iter_arg no body-level unpack/repack is emitted.
func.func @test_scf_while_memref_static(%cond: i1, %p: !fly.ptr<f32, shared>) {
  %l = fly.static : !fly.layout<4:1>
  %v0 = fly.make_view(%p, %l) : (!fly.ptr<f32, shared>, !fly.layout<4:1>) -> !fly.memref<f32, shared, 4:1>
  %r = scf.while (%a = %v0) : (!fly.memref<f32, shared, 4:1>) -> !fly.memref<f32, shared, 4:1> {
    scf.condition(%cond) %a : !fly.memref<f32, shared, 4:1>
  } do {
  ^bb0(%a: !fly.memref<f32, shared, 4:1>):
    scf.yield %a : !fly.memref<f32, shared, 4:1>
  }
  %sink = fly.get_iter(%r) : (!fly.memref<f32, shared, 4:1>) -> !fly.ptr<f32, shared>
  return
}

// -----

// CHECK-LABEL: func.func @test_scf_for_body_use_static(
// CHECK-SAME: %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: !llvm.ptr<3>)
// CHECK: return
// scf.for whose body actually USES the iter_arg as a memref: source
// materialization fires inside the body, rebuilding a memref via
// fly.make_view(iter_arg_ptr, static_layout) at the use site.
func.func @test_scf_for_body_use_static(%lb: index, %ub: index, %st: index, %p: !fly.ptr<f32, shared>) {
  %l = fly.static : !fly.layout<4:1>
  %v0 = fly.make_view(%p, %l) : (!fly.ptr<f32, shared>, !fly.layout<4:1>) -> !fly.memref<f32, shared, 4:1>
  %r = scf.for %i = %lb to %ub step %st iter_args(%a = %v0) -> (!fly.memref<f32, shared, 4:1>) {
    %ai = fly.get_iter(%a) : (!fly.memref<f32, shared, 4:1>) -> !fly.ptr<f32, shared>
    scf.yield %a : !fly.memref<f32, shared, 4:1>
  }
  return
}

// -----

// CHECK-LABEL: func.func @test_scf_if_memref_dynamic(
// CHECK-SAME: %{{.*}}: i1, %{{.*}}: !llvm.ptr<3>, %{{.*}}: !llvm.ptr<3>, %{{.*}}: i32)
// CHECK: return
// scf.if yielding a memref with dynamic layout: carrier is 1:N (ptr + layout
// struct). The scf.if op gains a second result, each branch yields two
// values, and the outer use rebuilds the memref via extractvalue +
// make_int_tuple + make_layout + make_view.
func.func @test_scf_if_memref_dynamic(%c: i1, %p0: !fly.ptr<f16, shared>, %p1: !fly.ptr<f16, shared>, %m: i32) {
  %sh = fly.make_int_tuple(%m) {static_args = array<i64: 16, -1>} : (i32) -> !fly.int_tuple<(16,?)>
  %sn = fly.make_int_tuple(%m) {static_args = array<i64: 1, -1>} : (i32) -> !fly.int_tuple<(1,?)>
  %l = fly.make_layout(%sh, %sn) : (!fly.int_tuple<(16,?)>, !fly.int_tuple<(1,?)>) -> !fly.layout<(16,?):(1,?)>
  %v0 = fly.make_view(%p0, %l) : (!fly.ptr<f16, shared>, !fly.layout<(16,?):(1,?)>) -> !fly.memref<f16, shared, (16,?):(1,?)>
  %v1 = fly.make_view(%p1, %l) : (!fly.ptr<f16, shared>, !fly.layout<(16,?):(1,?)>) -> !fly.memref<f16, shared, (16,?):(1,?)>
  %r = scf.if %c -> (!fly.memref<f16, shared, (16,?):(1,?)>) {
    scf.yield %v0 : !fly.memref<f16, shared, (16,?):(1,?)>
  } else {
    scf.yield %v1 : !fly.memref<f16, shared, (16,?):(1,?)>
  }
  %sink = fly.get_iter(%r) : (!fly.memref<f16, shared, (16,?):(1,?)>) -> !fly.ptr<f16, shared>
  return
}

// -----

// CHECK-LABEL: func.func @test_scf_for_body_use_dynamic(
// CHECK-SAME: %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: !llvm.ptr<3>, %{{.*}}: i32)
// CHECK: return
// scf.for whose body USES a dynamic-layout iter_arg: combines 1:N carrier
// expansion (ptr + layout struct) with in-body source materialization.
// Inside the body, the iter_arg ptr/struct pair is unpacked and rebuilt as a
// memref via extractvalue + make_int_tuple + make_layout + make_view before
// the use, then the original iter_arg carrier values are yielded back.
func.func @test_scf_for_body_use_dynamic(%lb: index, %ub: index, %st: index, %p: !fly.ptr<f16, shared>, %m: i32) {
  %sh = fly.make_int_tuple(%m) {static_args = array<i64: 16, -1>} : (i32) -> !fly.int_tuple<(16,?)>
  %sn = fly.make_int_tuple(%m) {static_args = array<i64: 1, -1>} : (i32) -> !fly.int_tuple<(1,?)>
  %l = fly.make_layout(%sh, %sn) : (!fly.int_tuple<(16,?)>, !fly.int_tuple<(1,?)>) -> !fly.layout<(16,?):(1,?)>
  %v0 = fly.make_view(%p, %l) : (!fly.ptr<f16, shared>, !fly.layout<(16,?):(1,?)>) -> !fly.memref<f16, shared, (16,?):(1,?)>
  %r = scf.for %i = %lb to %ub step %st iter_args(%a = %v0) -> (!fly.memref<f16, shared, (16,?):(1,?)>) {
    %ai = fly.get_iter(%a) : (!fly.memref<f16, shared, (16,?):(1,?)>) -> !fly.ptr<f16, shared>
    scf.yield %a : !fly.memref<f16, shared, (16,?):(1,?)>
  }
  return
}
