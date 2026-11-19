// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s | %fly-opt | FileCheck %s

// The assembly of each Fly type. Every signature is parsed, printed and parsed again; the CHECK
// lines pin the printed form, which drops whitespace, default fields and redundant spellings.

// A leaf is static, dynamic (with its width, divisibility and fast-divmod facts), a ratio, a
// basis or the empty mode `*`.
// CHECK-LABEL: func.func private @int_tuple_leaves(
// CHECK-SAME: !fly.int_tuple<-7>,
// CHECK-SAME: !fly.int_tuple<?>,
// CHECK-SAME: !fly.int_tuple<?{i64}>,
// CHECK-SAME: !fly.int_tuple<?{div=4}>,
// CHECK-SAME: !fly.int_tuple<?{i64 div=8 fdm}>,
// CHECK-SAME: !fly.int_tuple<?>,
// CHECK-SAME: !fly.int_tuple<3/4>,
// CHECK-SAME: !fly.int_tuple<2>,
// CHECK-SAME: !fly.int_tuple<2E0E1>,
// CHECK-SAME: !fly.int_tuple<?{div=2}E1>,
// CHECK-SAME: !fly.int_tuple<-1/2E0>,
// CHECK-SAME: !fly.int_tuple<*>)
func.func private @int_tuple_leaves(
    !fly.int_tuple<-7>, !fly.int_tuple<?>, !fly.int_tuple<?{i64}>, !fly.int_tuple<?{div=4}>,
    !fly.int_tuple<?{i64 div=8 fdm}>, !fly.int_tuple<?{i32 div=1}>, !fly.int_tuple<6/8>,
    !fly.int_tuple<4/2>, !fly.int_tuple<2E0E1>, !fly.int_tuple<?{div=2}E1>, !fly.int_tuple<-1/2E0>,
    !fly.int_tuple<*>)

// A rank-1 tuple is distinct from its leaf.
// CHECK-LABEL: func.func private @int_tuples(
// CHECK-SAME: !fly.int_tuple<()>,
// CHECK-SAME: !fly.int_tuple<(4)>,
// CHECK-SAME: !fly.int_tuple<((2,4),8)>,
// CHECK-SAME: !fly.int_tuple<(4,(?,1E1),*)>)
func.func private @int_tuples(
    !fly.int_tuple<()>, !fly.int_tuple<(4)>, !fly.int_tuple<( (2, 4), 8 )>, !fly.int_tuple<(4, (?, 1E1), *)>)

// CHECK-LABEL: func.func private @layouts(
// CHECK-SAME: !fly.layout<8:1>,
// CHECK-SAME: !fly.layout<(4):(1)>,
// CHECK-SAME: !fly.layout<((2,4),8):((1,2),8)>,
// CHECK-SAME: !fly.layout<(4,?):(1,?{i64})>,
// CHECK-SAME: !fly.layout<(4,8):(1E0,1E1)>,
// CHECK-SAME: !fly.layout<(4,8):(1/2,2)>)
func.func private @layouts(
    !fly.layout<8:1>, !fly.layout<(4) : (1)>, !fly.layout<((2, 4), 8) : ((1, 2), 8)>,
    !fly.layout<(4,?):(1,?{i64})>, !fly.layout<(4,8):(1E0,1E1)>, !fly.layout<(4,8):(1/2,2)>)

// CHECK-LABEL: func.func private @swizzles(
// CHECK-SAME: !fly.swizzle<S<3,3,3>>,
// CHECK-SAME: !fly.coord_swizzle<CS<3,0,[1],0,[2]>>,
// CHECK-SAME: !fly.coord_swizzle<CS<0,0,[],0,[]>>)
func.func private @swizzles(
    !fly.swizzle<S<3,3,3>>, !fly.coord_swizzle<CS<3,0,[1],0,[2]>>, !fly.coord_swizzle<CS<0,0,[],0,[]>>)

// A composed layout is inner o offset o outer. A composed inner extends the chain; a composed
// outer is bracketed.
// CHECK-LABEL: func.func private @composed_layouts(
// CHECK-SAME: !fly.composed_layout<S<3,3,3> o 0 o (8,8):(8,1)>,
// CHECK-SAME: !fly.composed_layout<CS<3,0,[1],0,[2]> o (0,0,0) o (4,8,8):(1E0,1E1,1E2)>,
// CHECK-SAME: !fly.composed_layout<32:2 o ?{div=4} o 8:4>,
// CHECK-SAME: !fly.composed_layout<CS<3,0,[1],0,[2]> o (0,0,0) o (4,8,8):(1E0,1E1,1E2) o (1,2,3) o (2,4,8):(1E0,1E1,1E2)>,
// CHECK-SAME: !fly.composed_layout<32:2 o 5 o [32:1 o 7 o (4,8):(1,4)]>)
func.func private @composed_layouts(
    !fly.composed_layout<S<3,3,3> o 0 o (8, 8) : (8, 1)>,
    !fly.composed_layout<CS<3,0,[1],0,[2]> o (0, 0, 0) o (4, 8, 8) : (1E0, 1E1, 1E2)>,
    !fly.composed_layout<32:2 o ?{div=4} o 8:4>,
    !fly.composed_layout<CS<3,0,[1],0,[2]> o (0,0,0) o (4,8,8):(1E0,1E1,1E2) o (1,2,3) o (2,4,8):(1E0,1E1,1E2)>,
    !fly.composed_layout<32:2 o 5 o [32:1 o 7 o (4,8):(1,4)]>)

// A tile mode is a layout, a scalar extent, `*`, or a bracketed list of modes.
// CHECK-LABEL: func.func private @tiles(
// CHECK-SAME: !fly.tile<*>,
// CHECK-SAME: !fly.tile<8>,
// CHECK-SAME: !fly.tile<[4|*]>,
// CHECK-SAME: !fly.tile<[?:1|(4,?{i64}):(1,4)|*]>,
// CHECK-SAME: !fly.tile<{{\[\[}}2:1|2:1]|4:1]>)
func.func private @tiles(
    !fly.tile<*>, !fly.tile<8>, !fly.tile<[4|*]>, !fly.tile<[?:1|(4,?{i64}):(1,4)|*]>,
    !fly.tile<[[2:1|2:1]|4:1]>)

// CHECK-LABEL: func.func private @scalars(
// CHECK-SAME: !fly.basis<2E0E1>,
// CHECK-SAME: !fly.fast_divmod<i64>)
func.func private @scalars(!fly.basis<2E0E1>, !fly.fast_divmod<i64>)

// The element's own alignment and the trivial swizzle are the defaults and are not printed.
// CHECK-LABEL: func.func private @pointers(
// CHECK-SAME: !fly.ptr<f32, global>,
// CHECK-SAME: !fly.ptr<f32, register>,
// CHECK-SAME: !fly.ptr<i8, global>,
// CHECK-SAME: !fly.ptr<f16, shared, align<16>>,
// CHECK-SAME: !fly.ptr<f16, shared, S<3,3,3>>,
// CHECK-SAME: !fly.ptr<f16, shared, align<16>, S<3,3,3>>)
func.func private @pointers(
    !fly.ptr<f32, global>, !fly.ptr<f32, register, align<4>>, !fly.ptr<i8, global, S<0,0,0>>,
    !fly.ptr<f16, shared, align<16>>, !fly.ptr<f16, shared, S<3,3,3>>, !fly.ptr<f16, shared, align<16>, S<3,3,3>>)

// CHECK-LABEL: func.func private @tensors(
// CHECK-SAME: !fly.memref<f32, global, (4,8):(1,4)>,
// CHECK-SAME: !fly.memref<f16, shared, S<3,3,3> o 0 o (8,8):(8,1), align<16>>,
// CHECK-SAME: !fly.coord_tensor<(0,0), (4,8):(1E0,1E1)>,
// CHECK-SAME: !fly.coord_tensor<?{i64}, 16:1>)
func.func private @tensors(
    !fly.memref<f32, global, (4, 8) : (1, 4)>, !fly.memref<f16, shared, S<3,3,3> o 0 o (8,8):(8,1), align<16>>,
    !fly.coord_tensor<(0, 0), (4, 8) : (1E0, 1E1)>, !fly.coord_tensor<?{i64}, 16:1>)

// The layouts of the tiled types print with their type prefix.
// CHECK-LABEL: func.func private @atoms(
// CHECK-SAME: !fly.copy_atom<!fly.universal_copy<64>, 16>,
// CHECK-SAME: !fly.copy_atom<!fly.universal_atomic<add, f32>, 32>,
// CHECK-SAME: !fly.mma_atom<!fly.universal_fma<1x1x1, (f32, f32) -> f32>>,
// CHECK-SAME: !fly.tiled_copy<!fly.copy_atom<!fly.universal_copy<32>, 32>, !fly.layout<(4,2):(2,1)>, !fly.tile<[2:1|4:1]>>,
// CHECK-SAME: !fly.tiled_mma<!fly.mma_atom<!fly.universal_fma<1x1x1, (f32, f32) -> f32>>, !fly.layout<(2,2,1):(1,2,0)>>,
// CHECK-SAME: !fly.tiled_mma<!fly.mma_atom<!fly.universal_fma<1x1x1, (f32, f32) -> f32>>, !fly.layout<(2,2,1):(1,2,0)>, !fly.tile<[4:1|*|*]>>)
func.func private @atoms(
    !fly.copy_atom<!fly.universal_copy<64>, 16>, !fly.copy_atom<!fly.universal_atomic<#fly<atomic_op add>, f32>, 32>,
    !fly.mma_atom<!fly.universal_fma<1x1x1, (f32, f32) -> f32>>,
    !fly.tiled_copy<!fly.copy_atom<!fly.universal_copy<32>, 32>, <(4,2):(2,1)>, <[2:1|4:1]>>,
    !fly.tiled_mma<!fly.mma_atom<!fly.universal_fma<1x1x1, (f32, f32) -> f32>>, <(2,2,1):(1,2,0)>>,
    !fly.tiled_mma<!fly.mma_atom<!fly.universal_fma<1x1x1, (f32, f32) -> f32>>, <(2,2,1):(1,2,0)>, <[4:1|*|*]>>)
