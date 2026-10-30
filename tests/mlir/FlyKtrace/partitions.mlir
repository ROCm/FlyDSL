// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

// RUN: %fly-opt %s --convert-fly-ktrace-to-rocdl --split-input-file --verify-diagnostics | FileCheck %s

// 64 partitions of a 64 MiB buffer: 65 cursor lines of 8 slots (520 slots), then 2096632 slots
// of wave regions, 8190 of them with a short last one. Wave k of partition p takes region
// k * 64 + p. A wave's partition is the low bits of its number, which counts the waves of a row
// of blocks consecutively; without a known block size, a block's wave count comes from its
// dimensions.
// CHECK-LABEL: gpu.func @partitioned
// CHECK: gpu.block_dim x
// CHECK: gpu.block_dim y
// CHECK: gpu.block_dim z
// CHECK: %[[WAVES:.*]] = arith.shrui
// CHECK: %[[BLOCK_X:.*]] = arith.index_cast %block_id_x{{[_0-9]*}} : index to i32
// CHECK-NEXT: arith.muli %[[BLOCK_X]], %[[WAVES]] : i32
// CHECK: %[[MASK:.*]] = llvm.mlir.constant(63 : i32)
// CHECK-NEXT: %[[PART:.*]] = arith.andi %{{.*}}, %[[MASK]] : i32
// CHECK: scf.if
// CHECK-DAG: %[[STRIDE:.*]] = llvm.mlir.constant(256 : i32)
// CHECK-DAG: %[[NEXT:.*]] = arith.addi %[[PART]], %{{.*}} : i32
// CHECK: arith.muli %[[NEXT]], %[[STRIDE]]
// CHECK: llvm.atomicrmw add
// CHECK: %[[WAVE:.*]] = rocdl.readfirstlane
// CHECK-DAG: %[[REGIONS:.*]] = llvm.mlir.constant(8190 : i32)
// CHECK-DAG: %[[P:.*]] = llvm.mlir.constant(64 : i32)
// CHECK: %[[K:.*]] = arith.minui %[[WAVE]], %[[REGIONS]]
// CHECK: %[[KP:.*]] = arith.muli %[[K]], %[[P]]
// CHECK: %[[INTERLEAVED:.*]] = arith.addi %[[KP]], %[[PART]]
// CHECK: arith.minui %[[INTERLEAVED]], %{{.*}}
// CHECK-DAG: llvm.mlir.constant(2096632 : i32)
// CHECK-DAG: llvm.mlir.constant(520 : i32)
// CHECK: llvm.ptrtoint %{{.*}} : !llvm.ptr<1> to i64

// A known block size gives the wave count as a constant: 4 for 256 threads.
// CHECK-LABEL: gpu.func @known_block
// CHECK: llvm.amdgcn.dispatch.id
// CHECK-NOT: gpu.block_dim
// CHECK: %[[WAVES:.*]] = llvm.mlir.constant(4 : i32)
// CHECK-NOT: gpu.block_dim
// CHECK: %[[BLOCK_X:.*]] = arith.index_cast %block_id_x{{[_0-9]*}} : index to i32
// CHECK-NEXT: arith.muli %[[BLOCK_X]], %[[WAVES]] : i32
// CHECK-NOT: gpu.block_dim
// CHECK: %[[MASK:.*]] = llvm.mlir.constant(63 : i32)
// CHECK-NEXT: arith.andi %{{.*}}, %[[MASK]] : i32
module attributes {fly_ktrace.capacity_slots = 2097151 : i32, fly_ktrace.partitions = 64 : i32, gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx950">] {
    gpu.func @partitioned(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "tick"
      gpu.return
    }
    gpu.func @known_block(%buf: !fly.ptr<i8, global>) kernel
        attributes {known_block_size = array<i32: 256, 1, 1>} {
      fly_ktrace.mark "tick"
      gpu.return
    }
  }
}

// -----

module attributes {fly_ktrace.partitions = 48 : i32, gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx950">] {
    // expected-error @below {{fly_ktrace.partitions must be a power of two, got 48}}
    gpu.func @not_a_power_of_two(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "tick"
      gpu.return
    }
  }
}

// -----

module attributes {fly_ktrace.capacity_slots = 127 : i32, fly_ktrace.partitions = 64 : i32, gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx950">] {
    // expected-error @below {{a 4096-byte trace buffer cannot hold 64 partitions}}
    gpu.func @too_small(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "tick"
      gpu.return
    }
  }
}
