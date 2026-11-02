// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-rocdl | FileCheck %s --implicit-check-not='!fly.ptr' --implicit-check-not=unrealized_conversion_cast
// RUN: %fly-opt %s --convert-scf-to-cf --convert-fly-to-rocdl | FileCheck %s --check-prefix=CF --implicit-check-not='!fly.ptr' --implicit-check-not=unrealized_conversion_cast

// Pointer carriers must be converted along with their consumers. In particular,
// canonicalizing a pointer-valued if can produce an arith.select.

// CHECK-LABEL: func.func @select_global
// CHECK-SAME: !llvm.ptr<1>
// CHECK: arith.select {{.*}} : !llvm.ptr<1>
// CHECK: llvm.load {{.*}} {alignment = 16 : i64} : !llvm.ptr<1> -> f32
// CF-LABEL: func.func @select_global
func.func @select_global(%c: i1, %a: !fly.ptr<f32, global, align<16>>, %b: !fly.ptr<f32, global, align<16>>) -> f32 {
  %p = arith.select %c, %a, %b : !fly.ptr<f32, global, align<16>>
  %v = fly.ptr.load(%p) : (!fly.ptr<f32, global, align<16>>) -> f32
  return %v : f32
}

// CHECK-LABEL: func.func @select_shared
// CHECK: arith.select {{.*}} : !llvm.ptr<3>
// CHECK: llvm.load {{.*}} {alignment = 8 : i64} : !llvm.ptr<3> -> f64
// CF-LABEL: func.func @select_shared
func.func @select_shared(%c: i1, %a: !fly.ptr<f64, shared, align<8>>, %b: !fly.ptr<f64, shared, align<8>>) -> f64 {
  %p = arith.select %c, %a, %b : !fly.ptr<f64, shared, align<8>>
  %v = fly.ptr.load(%p) : (!fly.ptr<f64, shared, align<8>>) -> f64
  return %v : f64
}

// Mixed address spaces in separate result slots must remain distinct.
// CHECK-LABEL: func.func @if_pointers
// CHECK: scf.if {{.*}} -> (!llvm.ptr<1>, !llvm.ptr<3>)
// CHECK: scf.yield {{.*}} : !llvm.ptr<1>, !llvm.ptr<3>
// CHECK: scf.yield {{.*}} : !llvm.ptr<1>, !llvm.ptr<3>
// CHECK: return {{.*}} : !llvm.ptr<1>, !llvm.ptr<3>
// CF-LABEL: func.func @if_pointers
// CF: cf.cond_br
// CF: cf.br {{.*}}!llvm.ptr<1>, !llvm.ptr<3>
func.func @if_pointers(%c: i1, %g: !fly.ptr<f32, global>, %s: !fly.ptr<f32, shared>) -> (!fly.ptr<f32, global>, !fly.ptr<f32, shared>) {
  %offset = fly.make_int_tuple() : () -> !fly.int_tuple<1>
  %p:2 = scf.if %c -> (!fly.ptr<f32, global>, !fly.ptr<f32, shared>) {
    %next = fly.add_offset(%g, %offset) : (!fly.ptr<f32, global>, !fly.int_tuple<1>) -> !fly.ptr<f32, global>
    scf.yield %next, %s : !fly.ptr<f32, global>, !fly.ptr<f32, shared>
  } else {
    %next = fly.add_offset(%s, %offset) : (!fly.ptr<f32, shared>, !fly.int_tuple<1>) -> !fly.ptr<f32, shared>
    scf.yield %g, %next : !fly.ptr<f32, global>, !fly.ptr<f32, shared>
  }
  return %p#0, %p#1 : !fly.ptr<f32, global>, !fly.ptr<f32, shared>
}

// A branch can yield poison to lanes that never read it, as KTRACE's split ifs do. A Fly
// pointer's poison takes the pointer's converted type; poison of a builtin type keeps its own.
// CHECK-LABEL: func.func @if_poison_pointers
// CHECK: } else {
// CHECK-DAG: %[[GLOBAL:.*]] = ub.poison : !llvm.ptr<1>
// CHECK-DAG: %[[BUFFER:.*]] = ub.poison : !llvm.struct<(ptr<8>, i32)>
// CHECK-DAG: %[[SMALL:.*]] = ub.poison : vector<4xf8E4M3FN>
// CHECK-DAG: %[[BITS:.*]] = arith.bitcast %[[SMALL]] : vector<4xf8E4M3FN> to vector<4xi8>
// CHECK: scf.yield %[[GLOBAL]], %[[BUFFER]], %[[BITS]] : !llvm.ptr<1>, !llvm.struct<(ptr<8>, i32)>, vector<4xi8>
// CF-LABEL: func.func @if_poison_pointers
// CF: ub.poison : !llvm.ptr<1>
// CF: cf.br {{.*}}!llvm.ptr<1>, !llvm.struct<(ptr<8>, i32)>, vector<4xi8>
func.func @if_poison_pointers(%c: i1, %g: !fly.ptr<f32, global>, %b: !fly.ptr<f32, #fly_rocdl.buffer_desc>, %v: vector<4xf8E4M3FN>)
    -> (!fly.ptr<f32, global>, !fly.ptr<f32, #fly_rocdl.buffer_desc>, vector<4xf8E4M3FN>) {
  %p:3 = scf.if %c -> (!fly.ptr<f32, global>, !fly.ptr<f32, #fly_rocdl.buffer_desc>, vector<4xf8E4M3FN>) {
    scf.yield %g, %b, %v : !fly.ptr<f32, global>, !fly.ptr<f32, #fly_rocdl.buffer_desc>, vector<4xf8E4M3FN>
  } else {
    %pg = ub.poison : !fly.ptr<f32, global>
    %pb = ub.poison : !fly.ptr<f32, #fly_rocdl.buffer_desc>
    %pv = ub.poison : vector<4xf8E4M3FN>
    scf.yield %pg, %pb, %pv : !fly.ptr<f32, global>, !fly.ptr<f32, #fly_rocdl.buffer_desc>, vector<4xf8E4M3FN>
  }
  return %p#0, %p#1, %p#2 : !fly.ptr<f32, global>, !fly.ptr<f32, #fly_rocdl.buffer_desc>, vector<4xf8E4M3FN>
}

// CHECK-LABEL: func.func @for_pointers
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (!llvm.ptr<1>, !llvm.ptr<3>)
// CHECK: llvm.getelementptr {{.*}} : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
// CHECK: llvm.getelementptr {{.*}} : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
// CHECK: scf.yield {{.*}} : !llvm.ptr<1>, !llvm.ptr<3>
// CF-LABEL: func.func @for_pointers
// CF: cf.br {{.*}}index, !llvm.ptr<1>, !llvm.ptr<3>
func.func @for_pointers(%n: index, %g: !fly.ptr<f32, global>, %s: !fly.ptr<f32, shared>) -> (!fly.ptr<f32, global>, !fly.ptr<f32, shared>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %offset = fly.make_int_tuple() : () -> !fly.int_tuple<1>
  %p:2 = scf.for %i = %zero to %n step %one iter_args(%gp = %g, %sp = %s) -> (!fly.ptr<f32, global>, !fly.ptr<f32, shared>) {
    %gn = fly.add_offset(%gp, %offset) : (!fly.ptr<f32, global>, !fly.int_tuple<1>) -> !fly.ptr<f32, global>
    %sn = fly.add_offset(%sp, %offset) : (!fly.ptr<f32, shared>, !fly.int_tuple<1>) -> !fly.ptr<f32, shared>
    scf.yield %gn, %sn : !fly.ptr<f32, global>, !fly.ptr<f32, shared>
  }
  return %p#0, %p#1 : !fly.ptr<f32, global>, !fly.ptr<f32, shared>
}

// CHECK-LABEL: func.func @while_pointer
// CHECK: scf.while {{.*}} : (!llvm.ptr<3>, i32) -> (!llvm.ptr<3>, i32)
// CHECK: scf.condition({{.*}}) {{.*}} : !llvm.ptr<3>, i32
// CHECK: ^bb0({{.*}}: !llvm.ptr<3>, {{.*}}: i32):
// CHECK: scf.yield {{.*}} : !llvm.ptr<3>, i32
// CF-LABEL: func.func @while_pointer
// CF: cf.cond_br
func.func @while_pointer(%n: i32, %p: !fly.ptr<i32, shared>) -> !fly.ptr<i32, shared> {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %offset = fly.make_int_tuple() : () -> !fly.int_tuple<1>
  %r:2 = scf.while (%ptr = %p, %i = %zero) : (!fly.ptr<i32, shared>, i32) -> (!fly.ptr<i32, shared>, i32) {
    %continue = arith.cmpi slt, %i, %n : i32
    scf.condition(%continue) %ptr, %i : !fly.ptr<i32, shared>, i32
  } do {
  ^bb0(%ptr: !fly.ptr<i32, shared>, %i: i32):
    %next = fly.add_offset(%ptr, %offset) : (!fly.ptr<i32, shared>, !fly.int_tuple<1>) -> !fly.ptr<i32, shared>
    %inc = arith.addi %i, %one : i32
    scf.yield %next, %inc : !fly.ptr<i32, shared>, i32
  }
  return %r#0 : !fly.ptr<i32, shared>
}

// A call/return boundary must not reintroduce the original Fly pointer type.
// CHECK-LABEL: func.func @call_pointer
// CHECK: call @if_pointers({{.*}}) : (i1, !llvm.ptr<1>, !llvm.ptr<3>) -> (!llvm.ptr<1>, !llvm.ptr<3>)
// CHECK: llvm.load {{.*}} : !llvm.ptr<1> -> f32
// CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> f32
// CF-LABEL: func.func @call_pointer
func.func @call_pointer(%c: i1, %g: !fly.ptr<f32, global>, %s: !fly.ptr<f32, shared>) -> f32 {
  %p:2 = func.call @if_pointers(%c, %g, %s) : (i1, !fly.ptr<f32, global>, !fly.ptr<f32, shared>) -> (!fly.ptr<f32, global>, !fly.ptr<f32, shared>)
  %a = fly.ptr.load(%p#0) : (!fly.ptr<f32, global>) -> f32
  %b = fly.ptr.load(%p#1) : (!fly.ptr<f32, shared>) -> f32
  %sum = arith.addf %a, %b : f32
  return %sum : f32
}
