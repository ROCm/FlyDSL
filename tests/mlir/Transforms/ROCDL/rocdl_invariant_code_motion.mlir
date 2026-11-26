// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --split-input-file --rocdl-invariant-code-motion | FileCheck %s

// A loop whose bounds are the same in every lane runs with the lanes active before it: an
// intrinsic that reads the active lanes moves out, and the LDS address an LDS DMA reads into M0
// is read once before it, after the address moves out.

// CHECK-LABEL: gpu.func @uniform_loop
// CHECK: %[[OFFSET:.*]] = arith.muli
// CHECK: %[[ADDR:.*]] = llvm.getelementptr %{{.*}}[64]
// CHECK-NEXT: %[[SOFFSET:.*]] = rocdl.readfirstlane %[[OFFSET]]
// CHECK-NEXT: %[[M0:.*]] = rocdl.readfirstlane %[[ADDR]]
// CHECK-NEXT: scf.for
// CHECK-NEXT: rocdl.raw.ptr.buffer.load.lds %{{.*}}, %[[M0]], %{{.*}}, %{{.*}}, %[[SOFFSET]],
gpu.module @kernels {
  gpu.func @uniform_loop(%rsrc: !llvm.ptr<8>, %lds: !llvm.ptr<3>, %voffset: i32, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %tid = gpu.thread_id x
    %t = arith.index_cast %tid : index to i32
    %offset = arith.muli %t, %c4 : i32
    %base = llvm.getelementptr %lds[%offset] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    scf.for %i = %c0 to %n step %c1 : i32 {
      %addr = llvm.getelementptr %base[64] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
      %soffset = rocdl.readfirstlane %offset : i32
      rocdl.raw.ptr.buffer.load.lds %rsrc, %addr, %c4, %voffset, %soffset, %c0, 0
    }
    gpu.return
  }
}

// -----

// Inner loops go first, so the M0 read placed before an inner loop moves on out of the outer
// one. The induction variable of such a loop is the same in every lane, and so is a loop bounded
// by it.

// CHECK-LABEL: gpu.func @nested_loops
// CHECK: %[[M0:.*]] = rocdl.readfirstlane %arg1
// CHECK-NEXT: scf.for %[[I:[a-z0-9_]+]] =
// CHECK-NEXT: scf.for %{{.*}} = %{{.*}} to %[[I]] step
// CHECK-NEXT: rocdl.raw.ptr.buffer.load.lds %{{.*}}, %[[M0]],
gpu.module @kernels {
  gpu.func @nested_loops(%rsrc: !llvm.ptr<8>, %lds: !llvm.ptr<3>, %voffset: i32, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    scf.for %i = %c0 to %n step %c1 : i32 {
      scf.for %j = %c0 to %i step %c1 : i32 {
        rocdl.raw.ptr.buffer.load.lds %rsrc, %lds, %c4, %voffset, %c0, %c0, 0
      }
    }
    gpu.return
  }
}

// -----

// Lanes run different numbers of iterations, so a lane active in the loop may be inactive
// before it. readlane and writelane, as ROCDL op or intrinsic call, name their lane, so they
// still move, unless that lane differs between lanes and is read from the first active one.

// CHECK-LABEL: gpu.func @divergent_loop
// CHECK: rocdl.readlane %{{.*}}, %c1_i32
// CHECK: llvm.call_intrinsic "llvm.amdgcn.writelane.i32"
// CHECK: scf.for
// CHECK-NEXT: rocdl.readfirstlane
// CHECK-NEXT: rocdl.readlane %{{.*}}, %{{.*}} :
// CHECK-NEXT: rocdl.raw.ptr.buffer.load.lds %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, 0
// CHECK-NEXT: rocdl.raw.ptr.buffer.load.lds %{{.*}}, %arg1,
gpu.module @kernels {
  gpu.func @divergent_loop(%rsrc: !llvm.ptr<8>, %lds: !llvm.ptr<3>, %voffset: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %tid = gpu.thread_id x
    %t = arith.index_cast %tid : index to i32
    scf.for %i = %c0 to %t step %c1 : i32 {
      %m0 = rocdl.readfirstlane %lds : !llvm.ptr<3>
      %own = rocdl.readlane %voffset, %t : (i32, i32) -> i32
      %soffset = rocdl.readlane %voffset, %c1 : (i32, i32) -> i32
      %vo = llvm.call_intrinsic "llvm.amdgcn.writelane.i32"(%soffset, %c0, %voffset) : (i32, i32, i32) -> i32
      rocdl.raw.ptr.buffer.load.lds %rsrc, %m0, %c4, %vo, %own, %c0, 0
      rocdl.raw.ptr.buffer.load.lds %rsrc, %lds, %c4, %vo, %own, %c0, 0
    }
    gpu.return
  }
}

// -----

// Inline assembly may depend on the lane whatever its operands: a loop bounded by it is not
// known to run the same iterations in every lane.

// CHECK-LABEL: gpu.func @inline_asm_bound
// CHECK: scf.for
// CHECK-NEXT: rocdl.readfirstlane
gpu.module @kernels {
  gpu.func @inline_asm_bound(%rsrc: !llvm.ptr<8>, %lds: !llvm.ptr<3>, %voffset: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %lane = llvm.inline_asm "v_mbcnt_lo_u32_b32 $0, -1, $1", "=v,v" %c0 : (i32) -> i32
    scf.for %i = %c0 to %lane step %c1 : i32 {
      %m0 = rocdl.readfirstlane %lds : !llvm.ptr<3>
      rocdl.raw.ptr.buffer.load.lds %rsrc, %m0, %c4, %voffset, %c0, %c0, 0
    }
    gpu.return
  }
}

// -----

// Inside a branch of the body, fewer lanes may be active than before the loop.

// CHECK-LABEL: gpu.func @under_branch
// CHECK: scf.for
// CHECK-NEXT: scf.if
// CHECK-NEXT: rocdl.readfirstlane
gpu.module @kernels {
  gpu.func @under_branch(%rsrc: !llvm.ptr<8>, %lds: !llvm.ptr<3>, %voffset: i32, %n: i32,
                         %cond: i1) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    scf.for %i = %c0 to %n step %c1 : i32 {
      scf.if %cond {
        %m0 = rocdl.readfirstlane %lds : !llvm.ptr<3>
        rocdl.raw.ptr.buffer.load.lds %rsrc, %m0, %c4, %voffset, %c0, %c0, 0
      }
    }
    gpu.return
  }
}

// -----

// An address that changes with the iteration stays in the loop, as does its M0 read.

// CHECK-LABEL: gpu.func @variant_address
// CHECK-NOT: rocdl.readfirstlane
// CHECK: scf.for
// CHECK-NEXT: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-NEXT: rocdl.readfirstlane %[[ADDR]]
// CHECK-NEXT: rocdl.raw.ptr.buffer.load.lds %{{.*}}, %{{.*}},
// CHECK-NEXT: rocdl.raw.ptr.buffer.load.lds %{{.*}}, %[[ADDR]],
gpu.module @kernels {
  gpu.func @variant_address(%rsrc: !llvm.ptr<8>, %lds: !llvm.ptr<3>, %voffset: i32, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    scf.for %i = %c0 to %n step %c1 : i32 {
      %addr = llvm.getelementptr %lds[%i] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
      %m0 = rocdl.readfirstlane %addr : !llvm.ptr<3>
      rocdl.raw.ptr.buffer.load.lds %rsrc, %m0, %c4, %voffset, %c0, %c0, 0
      rocdl.raw.ptr.buffer.load.lds %rsrc, %addr, %c4, %voffset, %c0, %c0, 0
    }
    gpu.return
  }
}

// -----

// A loop may not run, and dividing by zero is undefined behavior: a division only moves out
// when its divisor is known to be neither zero nor -1.

// CHECK-LABEL: gpu.func @division
// CHECK: arith.floordivsi %{{.*}}, %c4_i32
// CHECK: scf.for
// CHECK-NEXT: arith.floordivsi %arg1, %arg2
gpu.module @kernels {
  gpu.func @division(%out: !llvm.ptr<1>, %a: i32, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    scf.for %i = %c0 to %n step %c1 : i32 {
      %q = arith.floordivsi %a, %n : i32
      %r = arith.floordivsi %a, %c4 : i32
      %s = arith.addi %q, %r : i32
      llvm.store %s, %out : i32, !llvm.ptr<1>
    }
    gpu.return
  }
}
