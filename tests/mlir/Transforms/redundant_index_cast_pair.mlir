// RUN: %fly-opt %s --fly-layout-lowering | FileCheck %s

// A round trip through an intermediate at least 32 bits wide is redundant --
// but only when the value is provably narrow enough to survive it.  Values
// derived from launch coordinates and constants qualify.

// CHECK-LABEL: func.func @fold_from_thread_id
// CHECK-NOT:     arith.index_cast
// CHECK:         return
func.func @fold_from_thread_id() -> index {
  %tid = gpu.thread_id x
  %c63 = arith.constant 63 : index
  %m = arith.andi %tid, %c63 : index
  %0 = arith.index_cast %m : index to i32
  %1 = arith.index_cast %0 : i32 to index
  return %1 : index
}

// A chain of workgroup-relative coordinates stays small and folds.

// CHECK-LABEL: gpu.func @fold_arith_chain
// CHECK-NOT:     arith.index_cast
// CHECK:         gpu.return
gpu.module @arith_chain {
  gpu.func @fold_arith_chain(%out: memref<?xindex>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %tid = gpu.thread_id x
    %dim = gpu.block_dim x
    %c4 = arith.constant 4 : index
    %z = arith.constant 0 : index
    %a = arith.muli %dim, %c4 : index
    %b = arith.addi %a, %tid : index
    %s = arith.shrui %b, %c4 : index
    %0 = arith.index_cast %s : index to i32
    %1 = arith.index_cast %0 : i32 to index
    memref.store %1, %out[%z] : memref<?xindex>
    gpu.return
  }
}

// A grid-relative coordinate carries the grid limit (2^31-1), so scaling it
// leaves i32 range and the pair must be kept.  A bare `block_id` still folds,
// because the limit itself fits in a signed i32.

// CHECK-LABEL: gpu.func @fold_bare_block_id
// CHECK-NOT:     arith.index_cast
// CHECK:         gpu.return
gpu.module @bare_block_id {
  gpu.func @fold_bare_block_id(%out: memref<?xindex>) kernel
      attributes {known_grid_size = array<i32: 1024, 1, 1>} {
    %bid = gpu.block_id x
    %z = arith.constant 0 : index
    %0 = arith.index_cast %bid : index to i32
    %1 = arith.index_cast %0 : i32 to index
    memref.store %1, %out[%z] : memref<?xindex>
    gpu.return
  }
}

// CHECK-LABEL: func.func @keep_scaled_block_id
// CHECK:         arith.index_cast
// CHECK:         arith.index_cast
func.func @keep_scaled_block_id() -> index {
  %bid = gpu.block_id x
  %c4 = arith.constant 4 : index
  %s = arith.muli %bid, %c4 : index
  %0 = arith.index_cast %s : index to i32
  %1 = arith.index_cast %0 : i32 to index
  return %1 : index
}

// CHECK-LABEL: func.func @fold_i64_intermediate
// CHECK-NOT:     arith.index_cast
// CHECK:         return
func.func @fold_i64_intermediate() -> index {
  %tid = gpu.thread_id x
  %0 = arith.index_cast %tid : index to i64
  %1 = arith.index_cast %0 : i64 to index
  return %1 : index
}

// A value of unknown range must NOT be folded: an `index` is 64-bit, so a
// value above 2^31 would not survive the trip through i32.  This is the
// unsoundness llvm/llvm-project@8c81064169c5 fixed upstream.

// CHECK-LABEL: func.func @keep_unknown_arg
// CHECK:         arith.index_cast %arg0 : index to i32
// CHECK:         arith.index_cast %{{.*}} : i32 to index
func.func @keep_unknown_arg(%a: index) -> index {
  %0 = arith.index_cast %a : index to i32
  %1 = arith.index_cast %0 : i32 to index
  return %1 : index
}

// Likewise when an unknown value enters the arithmetic chain.

// CHECK-LABEL: func.func @keep_tainted_chain
// CHECK:         arith.index_cast
// CHECK:         arith.index_cast
func.func @keep_tainted_chain(%a: index) -> index {
  %tid = gpu.thread_id x
  %b = arith.addi %tid, %a : index
  %0 = arith.index_cast %b : index to i32
  %1 = arith.index_cast %0 : i32 to index
  return %1 : index
}

// A narrow intermediate truncates even a bounded value, so it is never folded.

// CHECK-LABEL: func.func @keep_i8
// CHECK:         arith.index_cast %{{.*}} : index to i8
// CHECK:         arith.index_cast %{{.*}} : i8 to index
func.func @keep_i8() -> index {
  %tid = gpu.thread_id x
  %0 = arith.index_cast %tid : index to i8
  %1 = arith.index_cast %0 : i8 to index
  return %1 : index
}

// CHECK-LABEL: func.func @keep_i16
// CHECK:         arith.index_cast %{{.*}} : index to i16
// CHECK:         arith.index_cast %{{.*}} : i16 to index
func.func @keep_i16() -> index {
  %tid = gpu.thread_id x
  %0 = arith.index_cast %tid : index to i16
  %1 = arith.index_cast %0 : i16 to index
  return %1 : index
}

// The unsigned round trip is left alone: the pattern is only instantiated for
// arith.index_cast, so index_castui is never rewritten even when the value is
// provably narrow.

// CHECK-LABEL: func.func @keep_unsigned_narrow
// CHECK:         arith.index_castui
// CHECK:         arith.index_castui
func.func @keep_unsigned_narrow() -> index {
  %tid = gpu.thread_id x
  %0 = arith.index_castui %tid : index to i32
  %1 = arith.index_castui %0 : i32 to index
  return %1 : index
}

// Vectors are handled through their element types.

// CHECK-LABEL: func.func @fold_vector
// CHECK-NOT:     arith.index_cast
// CHECK:         return
func.func @fold_vector() -> vector<4xindex> {
  %v = arith.constant dense<[1, 2, 3, 4]> : vector<4xindex>
  %0 = arith.index_cast %v : vector<4xindex> to vector<4xi32>
  %1 = arith.index_cast %0 : vector<4xi32> to vector<4xindex>
  return %1 : vector<4xindex>
}

// A mixed signed/unsigned pair is not a round trip at all, and the outer
// index_cast must not treat the inner index_castui as its own inverse.

// CHECK-LABEL: func.func @keep_mixed_signedness
// CHECK:         arith.index_castui
// CHECK:         arith.index_cast
func.func @keep_mixed_signedness() -> index {
  %tid = gpu.thread_id x
  %0 = arith.index_castui %tid : index to i32
  %1 = arith.index_cast %0 : i32 to index
  return %1 : index
}

// Unsigned casts are never folded, bounded operand or not.

// CHECK-LABEL: func.func @keep_unsigned_unknown
// CHECK:         arith.index_castui %arg0
// CHECK:         arith.index_castui
func.func @keep_unsigned_unknown(%a: index) -> index {
  %0 = arith.index_castui %a : index to i32
  %1 = arith.index_castui %0 : i32 to index
  return %1 : index
}

// Bounded roots are not enough on their own: arithmetic can carry a small value
// out of i32 range, and then the round trip is lossy again.

// CHECK-LABEL: func.func @keep_large_constant
// CHECK:         arith.index_cast
// CHECK:         arith.index_cast
func.func @keep_large_constant() -> index {
  %big = arith.constant 1099511627776 : index
  %tid = gpu.thread_id x
  %s = arith.addi %big, %tid : index
  %0 = arith.index_cast %s : index to i32
  %1 = arith.index_cast %0 : i32 to index
  return %1 : index
}

// CHECK-LABEL: func.func @keep_mul_overflow
// CHECK:         arith.index_cast
// CHECK:         arith.index_cast
func.func @keep_mul_overflow() -> index {
  %t = gpu.thread_id x
  %a = arith.muli %t, %t : index
  %b = arith.muli %a, %t : index
  %c = arith.muli %b, %t : index
  %0 = arith.index_cast %c : index to i32
  %1 = arith.index_cast %0 : i32 to index
  return %1 : index
}

// CHECK-LABEL: func.func @keep_shift_overflow
// CHECK:         arith.index_cast
// CHECK:         arith.index_cast
func.func @keep_shift_overflow() -> index {
  %t = gpu.thread_id x
  %c40 = arith.constant 40 : index
  %s = arith.shli %t, %c40 : index
  %0 = arith.index_cast %s : index to i32
  %1 = arith.index_cast %0 : i32 to index
  return %1 : index
}

// Shrinking operations keep a value in range, so these still fold.

// CHECK-LABEL: gpu.func @fold_masked
// CHECK-NOT:     arith.index_cast
// CHECK:         gpu.return
gpu.module @masked {
  gpu.func @fold_masked(%out: memref<?xindex>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %t = gpu.thread_id x
    %c63 = arith.constant 63 : index
    %big = arith.constant 1099511627776 : index
    %z = arith.constant 0 : index
    %m = arith.andi %big, %c63 : index
    %s = arith.addi %m, %t : index
    %0 = arith.index_cast %s : index to i32
    %1 = arith.index_cast %0 : i32 to index
    memref.store %1, %out[%z] : memref<?xindex>
    gpu.return
  }
}

// A negative constant is not a narrow value: every bound rule reads its
// operands as unsigned, where -1 is UINT64_MAX rather than 1.  The select
// keeps the constant from being folded away before the pattern runs.

// CHECK-LABEL: func.func @keep_negative_constant
// CHECK:         arith.index_cast
// CHECK:         arith.index_cast
func.func @keep_negative_constant(%p: i1) -> index {
  %cneg = arith.constant -1 : index
  %c1 = arith.constant 1 : index
  %c = arith.select %p, %cneg, %c1 : index
  %0 = arith.index_cast %c : index to i32
  %1 = arith.index_cast %0 : i32 to index
  return %1 : index
}

// Subtraction is not bounded by its operands: read unsigned, `tid - 1` wraps to
// UINT64_MAX when tid is 0, even though both sides are tiny.

// CHECK-LABEL: func.func @keep_subtraction
// CHECK:         arith.index_cast
// CHECK:         arith.index_cast
func.func @keep_subtraction() -> index {
  %c0 = gpu.thread_id x
  %c1 = arith.constant 1 : index
  %d = arith.subi %c0, %c1 : index
  %0 = arith.index_cast %d : index to i32
  %1 = arith.index_cast %0 : i32 to index
  return %1 : index
}

// A shift-right whose left operand came from a subtraction inherits that
// unknown range instead of the shift narrowing it.

// CHECK-LABEL: func.func @keep_shift_of_subtraction
// CHECK:         arith.index_cast
// CHECK:         arith.index_cast
func.func @keep_shift_of_subtraction(%a: index) -> index {
  %c1 = arith.constant 1 : index
  %d = arith.subi %c1, %a : index
  %s = arith.shrui %d, %c1 : index
  %0 = arith.index_cast %s : index to i32
  %1 = arith.index_cast %0 : i32 to index
  return %1 : index
}

// A declared upper_bound is believed over the built-in fallback, so a thread id
// scaled past what the fallback would allow still folds when the launch
// geometry says it is small.

// CHECK-LABEL: func.func @fold_declared_upper_bound
// CHECK-NOT:     arith.index_cast
// CHECK:         return
func.func @fold_declared_upper_bound() -> index {
  %tid = gpu.thread_id x upper_bound 256
  %c = arith.constant 65536 : index
  %m = arith.muli %tid, %c : index
  %0 = arith.index_cast %m : index to i32
  %1 = arith.index_cast %0 : i32 to index
  return %1 : index
}

// The same scaling is refused without the attribute: the fallback bound has to
// assume a far larger workgroup, and the product leaves i32 range.

// CHECK-LABEL: func.func @keep_scaled_without_bound
// CHECK:         arith.index_cast
// CHECK:         arith.index_cast
func.func @keep_scaled_without_bound() -> index {
  %tid = gpu.thread_id x
  %c = arith.constant 65536 : index
  %m = arith.muli %tid, %c : index
  %0 = arith.index_cast %m : index to i32
  %1 = arith.index_cast %0 : i32 to index
  return %1 : index
}

// A block id with a declared bound is likewise trusted over the grid fallback.

// CHECK-LABEL: func.func @fold_block_id_upper_bound
// CHECK-NOT:     arith.index_cast
// CHECK:         return
func.func @fold_block_id_upper_bound() -> index {
  %bid = gpu.block_id x upper_bound 1024
  %c = arith.constant 1024 : index
  %m = arith.muli %bid, %c : index
  %0 = arith.index_cast %m : index to i32
  %1 = arith.index_cast %0 : i32 to index
  return %1 : index
}

// The kernel's declared launch geometry is exact, and beats the fallback: with
// known_block_size the thread id is at most 256, so scaling by 2^20 stays well
// inside i32 even though the fallback bound would refuse it.

// CHECK-LABEL: gpu.func @fold_known_block_size
// CHECK-NOT:     arith.index_cast
// CHECK:         gpu.return
gpu.module @m {
  gpu.func @fold_known_block_size(%out: memref<?xindex>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %tid = gpu.thread_id x
    %c = arith.constant 1048576 : index
    %z = arith.constant 0 : index
    %m = arith.muli %tid, %c : index
    %0 = arith.index_cast %m : index to i32
    %1 = arith.index_cast %0 : i32 to index
    memref.store %1, %out[%z] : memref<?xindex>
    gpu.return
  }

  // Without the attribute the same kernel keeps the casts: the fallback has to
  // assume a workgroup far larger than 256, and the product leaves i32 range.

  // CHECK-LABEL: gpu.func @keep_without_known_block_size
  // CHECK:         arith.index_cast
  // CHECK:         arith.index_cast
  gpu.func @keep_without_known_block_size(%out: memref<?xindex>) kernel {
    %tid = gpu.thread_id x
    %c = arith.constant 1048576 : index
    %z = arith.constant 0 : index
    %m = arith.muli %tid, %c : index
    %0 = arith.index_cast %m : index to i32
    %1 = arith.index_cast %0 : i32 to index
    memref.store %1, %out[%z] : memref<?xindex>
    gpu.return
  }
}

// Widening is as dangerous as narrowing for the signed cast: whatever the
// narrow type already holds gets sign-extended, so a value that overflowed i8
// between the two casts comes back as a huge unsigned one.  The bound has to be
// checked against the narrower of the two types, not just the destination.

// CHECK-LABEL: func.func @keep_widening_after_i8_overflow
// CHECK:         arith.index_cast %{{.*}} : index to i32
// CHECK:         arith.index_cast %{{.*}} : i32 to index
func.func @keep_widening_after_i8_overflow() -> index {
  %tid = gpu.thread_id x
  %c127 = arith.constant 127 : index
  %m = arith.andi %tid, %c127 : index
  %b = arith.index_cast %m : index to i8
  %c100 = arith.constant 100 : i8
  %w = arith.addi %b, %c100 : i8
  %x = arith.index_cast %w : i8 to index
  %c1 = arith.constant 1 : index
  %s = arith.shrui %x, %c1 : index
  %0 = arith.index_cast %s : index to i32
  %1 = arith.index_cast %0 : i32 to index
  return %1 : index
}

// The same taint reaching divui, and the one-unknown-side rules: each of these
// would take the bound from the tainted operand alone.

// CHECK-LABEL: func.func @keep_divui_of_widened
// CHECK:         arith.index_cast %{{.*}} : index to i32
// CHECK:         arith.index_cast %{{.*}} : i32 to index
func.func @keep_divui_of_widened() -> index {
  %tid = gpu.thread_id x
  %c127 = arith.constant 127 : index
  %m = arith.andi %tid, %c127 : index
  %b = arith.index_cast %m : index to i8
  %c100 = arith.constant 100 : i8
  %w = arith.addi %b, %c100 : i8
  %x = arith.index_cast %w : i8 to index
  %c2 = arith.constant 2 : index
  %d = arith.divui %x, %c2 : index
  %0 = arith.index_cast %d : index to i32
  %1 = arith.index_cast %0 : i32 to index
  return %1 : index
}

// CHECK-LABEL: func.func @keep_minui_of_widened
// CHECK:         arith.index_cast %{{.*}} : index to i32
// CHECK:         arith.index_cast %{{.*}} : i32 to index
func.func @keep_minui_of_widened(%u: index) -> index {
  %tid = gpu.thread_id x
  %c127 = arith.constant 127 : index
  %m = arith.andi %tid, %c127 : index
  %b = arith.index_cast %m : index to i8
  %c100 = arith.constant 100 : i8
  %w = arith.addi %b, %c100 : i8
  %x = arith.index_cast %w : i8 to index
  %mn = arith.minui %u, %x : index
  %0 = arith.index_cast %mn : index to i32
  %1 = arith.index_cast %0 : i32 to index
  return %1 : index
}

// A value that stays inside i8's signed range survives the widening, so the
// round trip through i32 is still redundant.

// CHECK-LABEL: func.func @fold_widening_within_i8
// CHECK-NOT:     arith.index_cast %{{.*}} : index to i32
// CHECK:         return
func.func @fold_widening_within_i8() -> index {
  %tid = gpu.thread_id x
  %c63 = arith.constant 63 : index
  %m = arith.andi %tid, %c63 : index
  %b = arith.index_cast %m : index to i8
  %x = arith.index_cast %b : i8 to index
  %0 = arith.index_cast %x : index to i32
  %1 = arith.index_cast %0 : i32 to index
  return %1 : index
}

// A 64-bit-or-wider intermediate cannot lose anything, and must not trip the
// shift that computes the narrowness threshold.

// CHECK-LABEL: func.func @fold_i128_intermediate
// CHECK-NOT:     arith.index_cast
// CHECK:         return
func.func @fold_i128_intermediate(%a: index) -> index {
  %0 = arith.index_cast %a : index to i128
  %1 = arith.index_cast %0 : i128 to index
  return %1 : index
}
