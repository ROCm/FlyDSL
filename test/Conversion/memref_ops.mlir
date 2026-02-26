// RUN: %fly-opt %s --convert-fly-to-rocdl | FileCheck %s

// MemRef load/store lowering tests:
//   - Scalar: fly.memref.load/store -> llvm.getelementptr + llvm.load/store
//   - Vector: fly.memref.load_vec/store_vec -> llvm.load/store directly

// -----

// === Scalar Load/Store ===

// CHECK-LABEL: @test_memref_load
// CHECK-SAME: (%[[MEM:.*]]: !llvm.ptr<1>)
func.func @test_memref_load(%mem: !fly.memref<f32, global, 32:1>) -> f32 {
  %idx = fly.make_int_tuple() : () -> !fly.int_tuple<5>
  // CHECK: %[[C5:.*]] = arith.constant 5 : index
  // CHECK: %[[I64:.*]] = arith.index_cast %[[C5]] : index to i64
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEM]][%[[I64]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  // CHECK: %[[VAL:.*]] = llvm.load %[[GEP]] : !llvm.ptr<1> -> f32
  %val = fly.memref.load(%mem, %idx) : (!fly.memref<f32, global, 32:1>, !fly.int_tuple<5>) -> f32
  // CHECK: return %[[VAL]]
  return %val : f32
}

// CHECK-LABEL: @test_memref_store
// CHECK-SAME: (%[[MEM:.*]]: !llvm.ptr<1>, %[[VAL:.*]]: f32)
func.func @test_memref_store(%mem: !fly.memref<f32, global, 32:1>, %val: f32) {
  %idx = fly.make_int_tuple() : () -> !fly.int_tuple<3>
  // CHECK: %[[C3:.*]] = arith.constant 3 : index
  // CHECK: %[[I64:.*]] = arith.index_cast %[[C3]] : index to i64
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEM]][%[[I64]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  // CHECK: llvm.store %[[VAL]], %[[GEP]] : f32, !llvm.ptr<1>
  fly.memref.store(%val, %mem, %idx) : (f32, !fly.memref<f32, global, 32:1>, !fly.int_tuple<3>) -> ()
  return
}

// CHECK-LABEL: @test_memref_load_f16_shared
// CHECK-SAME: (%[[MEM:.*]]: !llvm.ptr<3>)
func.func @test_memref_load_f16_shared(%mem: !fly.memref<f16, shared, (16,32):(1,16)>) -> f16 {
  %idx = fly.make_int_tuple() : () -> !fly.int_tuple<10>
  // CHECK: %[[C10:.*]] = arith.constant 10 : index
  // CHECK: %[[I64:.*]] = arith.index_cast %[[C10]] : index to i64
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEM]][%[[I64]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16
  // CHECK: %[[VAL:.*]] = llvm.load %[[GEP]] : !llvm.ptr<3> -> f16
  %val = fly.memref.load(%mem, %idx) : (!fly.memref<f16, shared, (16,32):(1,16)>, !fly.int_tuple<10>) -> f16
  // CHECK: return %[[VAL]]
  return %val : f16
}

// -----

// === Vector Load/Store ===

// CHECK-LABEL: @test_load_vec
// CHECK-SAME: (%[[MEM:.*]]: !llvm.ptr<5>)
func.func @test_load_vec(%mem: !fly.memref<f32, register, 4:1>) -> vector<4xf32> {
  // CHECK: %[[VEC:.*]] = llvm.load %[[MEM]] : !llvm.ptr<5> -> vector<4xf32>
  %vec = fly.memref.load_vec(%mem) : (!fly.memref<f32, register, 4:1>) -> vector<4xf32>
  // CHECK: return %[[VEC]]
  return %vec : vector<4xf32>
}

// CHECK-LABEL: @test_store_vec
// CHECK-SAME: (%[[MEM:.*]]: !llvm.ptr<5>, %[[VEC:.*]]: vector<4xf32>)
func.func @test_store_vec(%mem: !fly.memref<f32, register, 4:1>, %vec: vector<4xf32>) {
  // CHECK: llvm.store %[[VEC]], %[[MEM]] : vector<4xf32>, !llvm.ptr<5>
  fly.memref.store_vec(%vec, %mem) : (vector<4xf32>, !fly.memref<f32, register, 4:1>) -> ()
  return
}

// -----

// === End-to-End: Alloca + Load + Store Pipeline ===

// CHECK-LABEL: @test_alloca_load_store
func.func @test_alloca_load_store() -> f32 {
  %s = fly.make_int_tuple() : () -> !fly.int_tuple<8>
  %d = fly.make_int_tuple() : () -> !fly.int_tuple<1>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<8>, !fly.int_tuple<1>) -> !fly.layout<8:1>
  // CHECK: %[[ALLOCA_SZ:.*]] = arith.constant 8 : i64
  // CHECK: %[[PTR:.*]] = llvm.alloca %[[ALLOCA_SZ]] x f32 : (i64) -> !llvm.ptr<5>
  %mem = fly.memref.alloca(%layout) : (!fly.layout<8:1>) -> !fly.memref<f32, register, 8:1>

  %idx_store = fly.make_int_tuple() : () -> !fly.int_tuple<2>
  %cst = arith.constant 42.0 : f32
  // CHECK: llvm.getelementptr
  // CHECK: llvm.store
  fly.memref.store(%cst, %mem, %idx_store) : (f32, !fly.memref<f32, register, 8:1>, !fly.int_tuple<2>) -> ()

  %idx_load = fly.make_int_tuple() : () -> !fly.int_tuple<2>
  // CHECK: llvm.getelementptr
  // CHECK: %[[LOADED:.*]] = llvm.load
  %val = fly.memref.load(%mem, %idx_load) : (!fly.memref<f32, register, 8:1>, !fly.int_tuple<2>) -> f32

  // CHECK: return %[[LOADED]]
  return %val : f32
}
