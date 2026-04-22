// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-rocdl | FileCheck %s

// MemRef load/store lowering tests:
//   - Scalar: fly.memref.load/store -> llvm.getelementptr + llvm.load/store
//   - Vector: fly.memref.load_vec/store_vec -> llvm.load/store directly

// -----

// === Scalar Load/Store ===

// CHECK-LABEL: @test_memref_load
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<1>)
func.func @test_memref_load(%mem: !fly.memref<f32, global, 32:1>) -> f32 {
  %idx = fly.make_int_tuple() : () -> !fly.int_tuple<5>
  // CHECK: %[[C5:.*]] = arith.constant 5 : i32
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[PTR]][%[[C5]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
  // CHECK: %[[VAL:.*]] = llvm.load %[[GEP]] : !llvm.ptr<1> -> f32
  %val = fly.memref.load(%mem, %idx) : (!fly.memref<f32, global, 32:1>, !fly.int_tuple<5>) -> f32
  // CHECK: return %[[VAL]]
  return %val : f32
}

// CHECK-LABEL: @test_memref_store
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<1>, %[[VAL:.*]]: f32)
func.func @test_memref_store(%mem: !fly.memref<f32, global, 32:1>, %val: f32) {
  %idx = fly.make_int_tuple() : () -> !fly.int_tuple<3>
  // CHECK: %[[C3:.*]] = arith.constant 3 : i32
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[PTR]][%[[C3]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
  // CHECK: llvm.store %[[VAL]], %[[GEP]] : f32, !llvm.ptr<1>
  fly.memref.store(%val, %mem, %idx) : (f32, !fly.memref<f32, global, 32:1>, !fly.int_tuple<3>) -> ()
  return
}

// CHECK-LABEL: @test_memref_load_f16_shared
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<3>)
func.func @test_memref_load_f16_shared(%mem: !fly.memref<f16, shared, (16,32):(1,16)>) -> f16 {
  %idx = fly.make_int_tuple() : () -> !fly.int_tuple<10>
  // CHECK: %[[C10:.*]] = arith.constant 10 : i32
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[PTR]][%[[C10]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
  // CHECK: %[[VAL:.*]] = llvm.load %[[GEP]] : !llvm.ptr<3> -> f16
  %val = fly.memref.load(%mem, %idx) : (!fly.memref<f16, shared, (16,32):(1,16)>, !fly.int_tuple<10>) -> f16
  // CHECK: return %[[VAL]]
  return %val : f16
}

// -----

// === Vector Load/Store ===
// load_vec/store_vec are lowered in LayoutLowering to vector-granularity
// PtrLoad/PtrStore (vec_width = first dim size) + vector.insert/extract_strided_slice,
// then PtrLoad/PtrStore become llvm.getelementptr + llvm.load/store.

// --- Contiguous (1D layout 4:1): single vector load/store ---

// CHECK-LABEL: @test_load_vec
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<5>)
func.func @test_load_vec(%mem: !fly.memref<f32, register, 4:1>) -> vector<4xf32> {
  // CHECK: %[[VEC:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf32>
  %vec = fly.memref.load_vec(%mem) : (!fly.memref<f32, register, 4:1>) -> vector<4xf32>
  // CHECK: return %[[VEC]]
  return %vec : vector<4xf32>
}

// CHECK-LABEL: @test_store_vec
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<5>, %[[VEC:.*]]: vector<4xf32>)
func.func @test_store_vec(%mem: !fly.memref<f32, register, 4:1>, %vec: vector<4xf32>) {
  // CHECK: llvm.store %[[VEC]], %{{.*}} : vector<4xf32>, !llvm.ptr<5>
  fly.memref.store_vec(%vec, %mem) : (vector<4xf32>, !fly.memref<f32, register, 4:1>) -> ()
  return
}

// --- 3D layout ((3,4,2):(1,16,4)): 8 chunks of vector<3>, column-major over (4,2) ---
// Column-major iteration over rest dims (4,2):
//   i=0 → coord=(0,0) → offset=0   i=1 → coord=(1,0) → offset=16
//   i=2 → coord=(2,0) → offset=32  i=3 → coord=(3,0) → offset=48
//   i=4 → coord=(0,1) → offset=4   i=5 → coord=(1,1) → offset=20
//   i=6 → coord=(2,1) → offset=36  i=7 → coord=(3,1) → offset=52

// CHECK-LABEL: @test_load_vec_3d
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<5>)
func.func @test_load_vec_3d(%mem: !fly.memref<f32, register, (3, 4, 2):(1, 16, 4)>) -> vector<24xf32> {
  // Chunk 0: offset=0
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: llvm.getelementptr %[[PTR]][%[[C0]]]
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> vector<3xf32>
  // CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [0]
  // Chunk 1: offset=16
  // CHECK: %[[C16:.*]] = arith.constant 16 : i32
  // CHECK: llvm.getelementptr %[[PTR]][%[[C16]]]
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> vector<3xf32>
  // CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [3]
  // Chunk 2: offset=32
  // CHECK: %[[C32:.*]] = arith.constant 32 : i32
  // CHECK: llvm.getelementptr %[[PTR]][%[[C32]]]
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> vector<3xf32>
  // CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [6]
  // Chunk 3: offset=48
  // CHECK: %[[C48:.*]] = arith.constant 48 : i32
  // CHECK: llvm.getelementptr %[[PTR]][%[[C48]]]
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> vector<3xf32>
  // CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [9]
  // Chunk 4: offset=4 (second iteration of dim-2)
  // CHECK: %[[C4:.*]] = arith.constant 4 : i32
  // CHECK: llvm.getelementptr %[[PTR]][%[[C4]]]
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> vector<3xf32>
  // CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [12]
  // Chunk 5: offset=20
  // CHECK: %[[C20:.*]] = arith.constant 20 : i32
  // CHECK: llvm.getelementptr %[[PTR]][%[[C20]]]
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> vector<3xf32>
  // CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [15]
  // Chunk 6: offset=36
  // CHECK: %[[C36:.*]] = arith.constant 36 : i32
  // CHECK: llvm.getelementptr %[[PTR]][%[[C36]]]
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> vector<3xf32>
  // CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [18]
  // Chunk 7: offset=52
  // CHECK: %[[C52:.*]] = arith.constant 52 : i32
  // CHECK: llvm.getelementptr %[[PTR]][%[[C52]]]
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> vector<3xf32>
  // CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [21]
  %vec = fly.memref.load_vec(%mem) : (!fly.memref<f32, register, (3, 4, 2):(1, 16, 4)>) -> vector<24xf32>
  return %vec : vector<24xf32>
}

// CHECK-LABEL: @test_store_vec_3d
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<5>, %[[VEC:.*]]: vector<24xf32>)
func.func @test_store_vec_3d(%mem: !fly.memref<f32, register, (3, 4, 2):(1, 16, 4)>, %vec: vector<24xf32>) {
  // Chunk 0: offset=0
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: llvm.getelementptr %[[PTR]][%[[C0]]]
  // CHECK: vector.extract_strided_slice %[[VEC]] {offsets = [0], sizes = [3]
  // CHECK: llvm.store %{{.*}}, %{{.*}} : vector<3xf32>, !llvm.ptr<5>
  // Chunk 4: offset=4 (verify column-major: dim-1 exhausted, dim-2 increments)
  // CHECK: arith.constant 16
  // CHECK: arith.constant 32
  // CHECK: arith.constant 48
  // CHECK: %[[C4:.*]] = arith.constant 4 : i32
  // CHECK: llvm.getelementptr %[[PTR]][%[[C4]]]
  // CHECK: vector.extract_strided_slice %[[VEC]] {offsets = [12], sizes = [3]
  // CHECK: llvm.store %{{.*}}, %{{.*}} : vector<3xf32>, !llvm.ptr<5>
  fly.memref.store_vec(%vec, %mem) : (vector<24xf32>, !fly.memref<f32, register, (3, 4, 2):(1, 16, 4)>) -> ()
  return
}

// --- Nested shape ((4,2),3):((1,4),16): first leaf dim = 4, 6 chunks of vector<4> ---
// Flattened shape: (4, 2, 3). vecWidth = 4 (first leaf, NOT product(4,2)=8).
// Column-major iteration over rest flat dims (2, 3):
//   i=0 → (0,0) → ((0,0),0) → offset=0    i=1 → (1,0) → ((0,1),0) → offset=4
//   i=2 → (0,1) → ((0,0),1) → offset=16   i=3 → (1,1) → ((0,1),1) → offset=20
//   i=4 → (0,2) → ((0,0),2) → offset=32   i=5 → (1,2) → ((0,1),2) → offset=36

// CHECK-LABEL: @test_load_vec_nested
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<5>)
func.func @test_load_vec_nested(%mem: !fly.memref<f32, register, ((4,2),3):((1,4),16)>) -> vector<24xf32> {
  // Chunk 0: offset=0
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: llvm.getelementptr %[[PTR]][%[[C0]]]
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf32>
  // CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [0]
  // Chunk 1: offset=4
  // CHECK: %[[C4:.*]] = arith.constant 4 : i32
  // CHECK: llvm.getelementptr %[[PTR]][%[[C4]]]
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf32>
  // CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [4]
  // Chunk 2: offset=16
  // CHECK: %[[C16:.*]] = arith.constant 16 : i32
  // CHECK: llvm.getelementptr %[[PTR]][%[[C16]]]
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf32>
  // CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [8]
  // Chunk 3: offset=20
  // CHECK: %[[C20:.*]] = arith.constant 20 : i32
  // CHECK: llvm.getelementptr %[[PTR]][%[[C20]]]
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf32>
  // CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [12]
  // Chunk 4: offset=32
  // CHECK: %[[C32:.*]] = arith.constant 32 : i32
  // CHECK: llvm.getelementptr %[[PTR]][%[[C32]]]
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf32>
  // CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [16]
  // Chunk 5: offset=36
  // CHECK: %[[C36:.*]] = arith.constant 36 : i32
  // CHECK: llvm.getelementptr %[[PTR]][%[[C36]]]
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf32>
  // CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [20]
  %vec = fly.memref.load_vec(%mem) : (!fly.memref<f32, register, ((4,2),3):((1,4),16)>) -> vector<24xf32>
  return %vec : vector<24xf32>
}

// CHECK-LABEL: @test_store_vec_nested
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<5>, %[[VEC:.*]]: vector<24xf32>)
func.func @test_store_vec_nested(%mem: !fly.memref<f32, register, ((4,2),3):((1,4),16)>, %vec: vector<24xf32>) {
  // Chunk 0: offset=0
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: llvm.getelementptr %[[PTR]][%[[C0]]]
  // CHECK: vector.extract_strided_slice %[[VEC]] {offsets = [0], sizes = [4]
  // CHECK: llvm.store %{{.*}}, %{{.*}} : vector<4xf32>, !llvm.ptr<5>
  // Chunk 3: offset=20 (verify column-major: dim-1 first then dim-2)
  // CHECK: arith.constant 4
  // CHECK: arith.constant 16
  // CHECK: %[[C20:.*]] = arith.constant 20 : i32
  // CHECK: llvm.getelementptr %[[PTR]][%[[C20]]]
  // CHECK: vector.extract_strided_slice %[[VEC]] {offsets = [12], sizes = [4]
  // CHECK: llvm.store %{{.*}}, %{{.*}} : vector<4xf32>, !llvm.ptr<5>
  fly.memref.store_vec(%vec, %mem) : (vector<24xf32>, !fly.memref<f32, register, ((4,2),3):((1,4),16)>) -> ()
  return
}


// CHECK-LABEL: @test_load_vec_stride1_dim1
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<5>)
func.func @test_load_vec_stride1_dim1(%mem: !fly.memref<f32, register, (2,4,3):(24,1,8)>) -> vector<24xf32> {
  // 6 chunk loads (vecWidth=4, numChunks=6), col-major over restShape (2,3):
  //   (0,0)→offset 0, (1,0)→offset 24, (0,1)→offset 8, (1,1)→offset 32, (0,2)→offset 16, (1,2)→offset 40
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf32>
  // CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [0]
  // CHECK: arith.constant 24
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf32>
  // CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [4]
  // CHECK: arith.constant 8
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf32>
  // CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [8]
  // CHECK: arith.constant 32
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf32>
  // CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [12]
  // CHECK: arith.constant 16
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf32>
  // CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [16]
  // CHECK: arith.constant 40
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf32>
  // CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [20]
  // Permute: shape_cast [24]→[3,2,4], transpose [0,2,1]→[3,4,2], shape_cast→[24]
  // CHECK: vector.shape_cast %{{.*}} : vector<24xf32> to vector<3x2x4xf32>
  // CHECK: vector.transpose %{{.*}}, [0, 2, 1] : vector<3x2x4xf32> to vector<3x4x2xf32>
  // CHECK: vector.shape_cast %{{.*}} : vector<3x4x2xf32> to vector<24xf32>
  %vec = fly.memref.load_vec(%mem) : (!fly.memref<f32, register, (2,4,3):(24,1,8)>) -> vector<24xf32>
  return %vec : vector<24xf32>
}

// CHECK-LABEL: @test_store_vec_stride1_dim1
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<5>, %[[VEC:.*]]: vector<24xf32>)
func.func @test_store_vec_stride1_dim1(%mem: !fly.memref<f32, register, (2,4,3):(24,1,8)>, %vec: vector<24xf32>) {
  // Reverse permute: shape_cast [24]→[3,4,2], transpose [0,2,1]→[3,2,4], shape_cast→[24]
  // CHECK: vector.shape_cast %[[VEC]] : vector<24xf32> to vector<3x4x2xf32>
  // CHECK: vector.transpose %{{.*}}, [0, 2, 1] : vector<3x4x2xf32> to vector<3x2x4xf32>
  // CHECK: %[[PERM:.*]] = vector.shape_cast %{{.*}} : vector<3x2x4xf32> to vector<24xf32>
  // 6 chunk stores
  // CHECK: arith.constant 0
  // CHECK: vector.extract_strided_slice %[[PERM]] {offsets = [0], sizes = [4]
  // CHECK: llvm.store %{{.*}}, %{{.*}} : vector<4xf32>, !llvm.ptr<5>
  // CHECK: arith.constant 24
  // CHECK: vector.extract_strided_slice %[[PERM]] {offsets = [4], sizes = [4]
  // CHECK: llvm.store %{{.*}}, %{{.*}} : vector<4xf32>, !llvm.ptr<5>
  // CHECK: arith.constant 8
  // CHECK: vector.extract_strided_slice %[[PERM]] {offsets = [8], sizes = [4]
  // CHECK: llvm.store %{{.*}}, %{{.*}} : vector<4xf32>, !llvm.ptr<5>
  // CHECK: arith.constant 32
  // CHECK: vector.extract_strided_slice %[[PERM]] {offsets = [12], sizes = [4]
  // CHECK: llvm.store %{{.*}}, %{{.*}} : vector<4xf32>, !llvm.ptr<5>
  // CHECK: arith.constant 16
  // CHECK: vector.extract_strided_slice %[[PERM]] {offsets = [16], sizes = [4]
  // CHECK: llvm.store %{{.*}}, %{{.*}} : vector<4xf32>, !llvm.ptr<5>
  // CHECK: arith.constant 40
  // CHECK: vector.extract_strided_slice %[[PERM]] {offsets = [20], sizes = [4]
  // CHECK: llvm.store %{{.*}}, %{{.*}} : vector<4xf32>, !llvm.ptr<5>
  fly.memref.store_vec(%vec, %mem) : (vector<24xf32>, !fly.memref<f32, register, (2,4,3):(24,1,8)>) -> ()
  return
}

// --- Scalar mode: no stride=1, (2,3):(4,8) → 6 scalar load/store ---
// Column-major iteration: (0,0)→0, (1,0)→4, (0,1)→8, (1,1)→12, (0,2)→16, (1,2)→20

// CHECK-LABEL: @test_load_vec_scalar
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<5>)
func.func @test_load_vec_scalar(%mem: !fly.memref<f32, register, (2,3):(4,8)>) -> vector<6xf32> {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: llvm.getelementptr %[[PTR]][%[[C0]]]
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> f32
  // CHECK: vector.insert %{{.*}}, %{{.*}} [0]
  // CHECK: arith.constant 4
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> f32
  // CHECK: vector.insert %{{.*}}, %{{.*}} [1]
  // CHECK: arith.constant 8
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> f32
  // CHECK: vector.insert %{{.*}}, %{{.*}} [2]
  // CHECK: arith.constant 12
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> f32
  // CHECK: vector.insert %{{.*}}, %{{.*}} [3]
  // CHECK: arith.constant 16
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> f32
  // CHECK: vector.insert %{{.*}}, %{{.*}} [4]
  // CHECK: arith.constant 20
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> f32
  // CHECK: vector.insert %{{.*}}, %{{.*}} [5]
  %vec = fly.memref.load_vec(%mem) : (!fly.memref<f32, register, (2,3):(4,8)>) -> vector<6xf32>
  return %vec : vector<6xf32>
}

// CHECK-LABEL: @test_store_vec_scalar
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<5>, %[[VEC:.*]]: vector<6xf32>)
func.func @test_store_vec_scalar(%mem: !fly.memref<f32, register, (2,3):(4,8)>, %vec: vector<6xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: llvm.getelementptr %[[PTR]][%[[C0]]]
  // CHECK: vector.extract %[[VEC]][0]
  // CHECK: llvm.store %{{.*}}, %{{.*}} : f32, !llvm.ptr<5>
  // CHECK: arith.constant 4
  // CHECK: vector.extract %[[VEC]][1]
  // CHECK: llvm.store %{{.*}}, %{{.*}} : f32, !llvm.ptr<5>
  // CHECK: arith.constant 8
  // CHECK: vector.extract %[[VEC]][2]
  // CHECK: llvm.store %{{.*}}, %{{.*}} : f32, !llvm.ptr<5>
  // CHECK: arith.constant 12
  // CHECK: vector.extract %[[VEC]][3]
  // CHECK: llvm.store %{{.*}}, %{{.*}} : f32, !llvm.ptr<5>
  // CHECK: arith.constant 16
  // CHECK: vector.extract %[[VEC]][4]
  // CHECK: llvm.store %{{.*}}, %{{.*}} : f32, !llvm.ptr<5>
  // CHECK: arith.constant 20
  // CHECK: vector.extract %[[VEC]][5]
  // CHECK: llvm.store %{{.*}}, %{{.*}} : f32, !llvm.ptr<5>
  fly.memref.store_vec(%vec, %mem) : (vector<6xf32>, !fly.memref<f32, register, (2,3):(4,8)>) -> ()
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
  // CHECK: %[[GEP:.*]] = llvm.getelementptr
  // CHECK: llvm.store
  fly.memref.store(%cst, %mem, %idx_store) : (f32, !fly.memref<f32, register, 8:1>, !fly.int_tuple<2>) -> ()

  %idx_load = fly.make_int_tuple() : () -> !fly.int_tuple<2>
  // CHECK: %[[LOADED:.*]] = llvm.load %[[GEP]]
  %val = fly.memref.load(%mem, %idx_load) : (!fly.memref<f32, register, 8:1>, !fly.int_tuple<2>) -> f32

  // CHECK: return %[[LOADED]]
  return %val : f32
}
