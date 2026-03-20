// RUN: %fly-opt %s --fly-rewrite-func-signature | FileCheck %s

// === Sink static DSL args from func ===

// Static int_tuple arg is removed; fly.static materializes it inside the body.
// CHECK-LABEL: @test_sink_static_int_tuple
// CHECK-SAME: ()
func.func @test_sink_static_int_tuple(%arg0: !fly.int_tuple<(4,8)>) {
  // CHECK: fly.static : !fly.int_tuple<(4,8)>
  return
}

// -----

// Static layout arg is removed.
// CHECK-LABEL: @test_sink_static_layout
// CHECK-SAME: ()
func.func @test_sink_static_layout(%arg0: !fly.layout<(4,8):(1,4)>) {
  // CHECK: fly.static : !fly.layout<(4,8):(1,4)>
  return
}

// -----

// Mixed: static DSL arg removed, non-DSL arg kept.
// CHECK-LABEL: @test_sink_mixed
// CHECK-SAME: (%{{.*}}: i32)
func.func @test_sink_mixed(%arg0: !fly.int_tuple<5>, %arg1: i32) {
  // CHECK: fly.static : !fly.int_tuple<5>
  return
}

// -----

// === Rewrite dynamic DSL args to LLVM struct ===

// Dynamic int_tuple arg becomes packed struct of its dynamic leaves.
// CHECK-LABEL: @test_dynamic_int_tuple
// CHECK-SAME: (%[[S:.*]]: !llvm.struct<packed (i32)>)
func.func @test_dynamic_int_tuple(%arg0: !fly.int_tuple<?>) {
  // CHECK: llvm.extractvalue %[[S]][0]
  // CHECK: fly.make_int_tuple
  return
}

// -----

// Nested dynamic int_tuple: two dynamic leaves -> struct with two i32 fields.
// CHECK-LABEL: @test_dynamic_nested_int_tuple
// CHECK-SAME: (%[[S:.*]]: !llvm.struct<packed (i32, i32)>)
func.func @test_dynamic_nested_int_tuple(%arg0: !fly.int_tuple<(?,?)>) {
  // CHECK: llvm.extractvalue %[[S]][0]
  // CHECK: llvm.extractvalue %[[S]][1]
  // CHECK: fly.make_int_tuple
  return
}

// -----

// Layout with dynamic shape and stride: struct contains two sub-structs.
// CHECK-LABEL: @test_dynamic_layout
// CHECK-SAME: (%[[S:.*]]: !llvm.struct<packed (struct<packed (i32)>, struct<packed (i32)>)>)
func.func @test_dynamic_layout(%arg0: !fly.layout<?:?>) {
  // CHECK: llvm.extractvalue %[[S]][0]
  // CHECK: fly.make_int_tuple
  // CHECK: llvm.extractvalue %[[S]][1]
  // CHECK: fly.make_int_tuple
  // CHECK: fly.make_layout
  return
}

// -----

// Layout with only dynamic stride; shape is static -> struct has one sub-struct.
// CHECK-LABEL: @test_partially_dynamic_layout
// CHECK-SAME: (%[[S:.*]]: !llvm.struct<packed (struct<packed (i32)>)>)
func.func @test_partially_dynamic_layout(%arg0: !fly.layout<4:?>) {
  // CHECK: fly.make_int_tuple{{.*}}() : () -> !fly.int_tuple<4>
  // CHECK: fly.make_int_tuple{{.*}} -> !fly.int_tuple<?>
  // CHECK: fly.make_layout
  return
}

// -----

// MemRef with static layout: lowered to a single fly.ptr argument.
// CHECK-LABEL: @test_static_memref
// CHECK-SAME: (%[[P:.*]]: !fly.ptr<f32, global>)
func.func @test_static_memref(%arg0: !fly.memref<f32, global, 32:1>) {
  // CHECK: fly.static : !fly.layout<32:1>
  // CHECK: fly.make_view(%[[P]]
  return
}

// -----

// MemRef with dynamic layout: lowered to ptr arg + layout struct arg.
// CHECK-LABEL: @test_dynamic_memref
// CHECK-SAME: (%[[P:.*]]: !fly.ptr<f16, shared>, %[[L:.*]]: !llvm.struct<packed (struct<packed (i32)>, struct<packed (i32)>)>)
func.func @test_dynamic_memref(%arg0: !fly.memref<f16, shared, (16,?):(1,?)>) {
  // CHECK: llvm.extractvalue %[[L]][0]
  // CHECK: llvm.extractvalue %[[L]][1]
  // CHECK: fly.make_layout
  // CHECK: fly.make_view(%[[P]]
  return
}

// -----

// Non-DSL args are passed through unchanged.
// CHECK-LABEL: @test_passthrough
// CHECK-SAME: (%{{.*}}: i32, %{{.*}}: f32)
func.func @test_passthrough(%arg0: i32, %arg1: f32) {
  return
}

// -----

// Declaration without body: signature is rewritten but no unpack code is generated.
// CHECK-LABEL: @test_declaration_dynamic
// CHECK-SAME: (!llvm.struct<packed (i32, i32)>)
func.func private @test_declaration_dynamic(!fly.int_tuple<(?,?)>)

// -----

// Declaration with static arg: arg is removed entirely.
// CHECK-LABEL: @test_declaration_static
// CHECK-SAME: ()
func.func private @test_declaration_static(!fly.int_tuple<(4,8)>)

// -----

// === gpu.launch_func: pack DSL operands + drop static operands ===

module attributes {gpu.container_module} {

gpu.module @kernel_mod_static {
  // CHECK-LABEL: gpu.func @static_kernel
  // CHECK-SAME: ()
  gpu.func @static_kernel(%arg0: !fly.int_tuple<(4,8)>) kernel {
    // CHECK: fly.static : !fly.int_tuple<(4,8)>
    gpu.return
  }
}

// Static operands are dropped from gpu.launch_func.
// CHECK-LABEL: @test_launch_static
func.func @test_launch_static(%t: !fly.int_tuple<(4,8)>) {
  %c1 = arith.constant 1 : index
  // CHECK: gpu.launch_func @kernel_mod_static::@static_kernel
  // CHECK-NOT: args(
  gpu.launch_func @kernel_mod_static::@static_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%t : !fly.int_tuple<(4,8)>)
  return
}

}

// -----

module attributes {gpu.container_module} {

gpu.module @kernel_mod_dynamic {
  // CHECK-LABEL: gpu.func @dynamic_kernel
  // CHECK-SAME: (%{{.*}}: !llvm.struct<packed (i32, i32)>)
  gpu.func @dynamic_kernel(%arg0: !fly.int_tuple<(?,?)>) kernel {
    // CHECK: llvm.extractvalue
    // CHECK: fly.make_int_tuple
    gpu.return
  }
}

// Dynamic operands are packed into structs at the launch_func call site.
// CHECK-LABEL: @test_launch_dynamic
func.func @test_launch_dynamic(%a: i32, %b: i32) {
  %t = fly.make_int_tuple(%a, %b) : (i32, i32) -> !fly.int_tuple<(?,?)>
  %c1 = arith.constant 1 : index
  // CHECK: llvm.mlir.undef : !llvm.struct<packed (i32, i32)>
  // CHECK: llvm.insertvalue
  // CHECK: llvm.insertvalue
  // CHECK: gpu.launch_func @kernel_mod_dynamic::@dynamic_kernel
  // CHECK-SAME: args(%{{.*}} : !llvm.struct<packed (i32, i32)>)
  gpu.launch_func @kernel_mod_dynamic::@dynamic_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%t : !fly.int_tuple<(?,?)>)
  return
}

}
