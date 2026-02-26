// RUN: %fly-opt %s --convert-fly-to-rocdl | FileCheck %s

// MemRefAlloca lowering tests:
//   fly.memref.alloca -> llvm.alloca with cosize as allocation count

// CHECK-LABEL: @test_memref_alloca
func.func @test_memref_alloca() {
  %s = fly.make_int_tuple() : () -> !fly.int_tuple<(4, 8)>
  %d = fly.make_int_tuple() : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  // Cosize of (4,8):(1,4) = max(4*1, 8*4) = 32
  // CHECK: %[[SIZE:.*]] = arith.constant 32 : i64
  // CHECK: llvm.alloca %[[SIZE]] x f32 : (i64) -> !llvm.ptr<5>
  %mem = fly.memref.alloca(%layout) : (!fly.layout<(4, 8) : (1, 4)>) -> !fly.memref<f32, register, (4, 8) : (1, 4)>
  return
}

// CHECK-LABEL: @test_memref_alloca_1d
func.func @test_memref_alloca_1d() {
  %s = fly.make_int_tuple() : () -> !fly.int_tuple<8>
  %d = fly.make_int_tuple() : () -> !fly.int_tuple<1>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<8>, !fly.int_tuple<1>) -> !fly.layout<8:1>
  // CHECK: %[[SIZE:.*]] = arith.constant 8 : i64
  // CHECK: llvm.alloca %[[SIZE]] x f32 : (i64) -> !llvm.ptr<5>
  %mem = fly.memref.alloca(%layout) : (!fly.layout<8:1>) -> !fly.memref<f32, register, 8:1>
  return
}
