// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --pass-pipeline='builtin.module(gpu.module(fly-attach-lds-alias-scope))' | FileCheck %s

// fly-attach-lds-alias-scope finds external `[0 x i8] addrspace(3)`
// LDS globals in a gpu.module, gives each one a distinct alias scope,
// and tags every load / store / amdgcn.raw.ptr.buffer.load.lds whose
// addrspace(3) pointer can be traced back to a single global through
// addressof / ptrtoint / add / inttoptr / GEP.

// -----------------------------------------------------------------------------
// Two named dyn-shared globals -> per-symbol alias_scopes / noalias_scopes on
// loads, with the int-derived pointer being recognised through
// ptrtoint+add+inttoptr.
// -----------------------------------------------------------------------------

// CHECK-DAG: #[[DOMAIN:.+]] = #llvm.alias_scope_domain<{{.*}}description = "FlyDynSharedDomain">
// CHECK-DAG: #[[SCOPE_A:.+]] = #llvm.alias_scope<{{.*}}domain = #[[DOMAIN]], description = "buf_a">
// CHECK-DAG: #[[SCOPE_B:.+]] = #llvm.alias_scope<{{.*}}domain = #[[DOMAIN]], description = "buf_b">

// CHECK-LABEL: gpu.module @two_named
gpu.module @two_named {
  llvm.mlir.global external @buf_a() {addr_space = 3 : i32, alignment = 1024 : i64, dso_local} : !llvm.array<0 x i8>
  llvm.mlir.global external @buf_b() {addr_space = 3 : i32, alignment = 1024 : i64, dso_local} : !llvm.array<0 x i8>

  // CHECK-LABEL: llvm.func @load_pair
  llvm.func @load_pair(%off: i32) -> vector<4xi32> {
    %a_ptr = llvm.mlir.addressof @buf_a : !llvm.ptr<3>
    %b_ptr = llvm.mlir.addressof @buf_b : !llvm.ptr<3>
    %a_int = llvm.ptrtoint %a_ptr : !llvm.ptr<3> to i32
    %b_int = llvm.ptrtoint %b_ptr : !llvm.ptr<3> to i32
    %a_off = llvm.add %a_int, %off : i32
    %b_off = llvm.add %b_int, %off : i32
    %a_p = llvm.inttoptr %a_off : i32 to !llvm.ptr<3>
    %b_p = llvm.inttoptr %b_off : i32 to !llvm.ptr<3>
    // CHECK: llvm.load %{{.+}} {alias_scopes = [#[[SCOPE_A]]], noalias_scopes = [#[[SCOPE_B]]]}
    %va = llvm.load %a_p : !llvm.ptr<3> -> vector<4xi32>
    // CHECK: llvm.load %{{.+}} {alias_scopes = [#[[SCOPE_B]]], noalias_scopes = [#[[SCOPE_A]]]}
    %vb = llvm.load %b_p : !llvm.ptr<3> -> vector<4xi32>
    %sum = llvm.add %va, %vb : vector<4xi32>
    llvm.return %sum : vector<4xi32>
  }
}

// -----

// -----------------------------------------------------------------------------
// Single-global module: pass is a no-op. Tagging a single scope gives the
// SI Wait Counter pass nothing extra to disambiguate.
// -----------------------------------------------------------------------------

// CHECK-LABEL: gpu.module @one_named
gpu.module @one_named {
  llvm.mlir.global external @only_buf() {addr_space = 3 : i32, alignment = 1024 : i64, dso_local} : !llvm.array<0 x i8>

  // CHECK-LABEL: llvm.func @load_only
  llvm.func @load_only() -> vector<4xi32> {
    %p = llvm.mlir.addressof @only_buf : !llvm.ptr<3>
    // CHECK: llvm.load
    // CHECK-NOT: alias_scopes
    %v = llvm.load %p : !llvm.ptr<3> -> vector<4xi32>
    llvm.return %v : vector<4xi32>
  }
}

// -----

// -----------------------------------------------------------------------------
// Static [N x i8] LDS globals (N > 0, the SmemAllocator pattern) are skipped.
// Their alias info already comes from distinct LLVM symbols.
// -----------------------------------------------------------------------------

// CHECK-LABEL: gpu.module @static_lds
gpu.module @static_lds {
  llvm.mlir.global external @smem_a() {addr_space = 3 : i32, alignment = 1024 : i64, dso_local} : !llvm.array<4096 x i8>
  llvm.mlir.global external @smem_b() {addr_space = 3 : i32, alignment = 1024 : i64, dso_local} : !llvm.array<4096 x i8>

  // CHECK-LABEL: llvm.func @load_static
  llvm.func @load_static() -> vector<4xi32> {
    %p = llvm.mlir.addressof @smem_a : !llvm.ptr<3>
    // CHECK: llvm.load
    // CHECK-NOT: alias_scopes
    %v = llvm.load %p : !llvm.ptr<3> -> vector<4xi32>
    llvm.return %v : vector<4xi32>
  }
}

// -----

// -----------------------------------------------------------------------------
// Ambiguous provenance: an `add` whose lhs is `ptrtoint(@A)` and rhs is
// `ptrtoint(@B)` produces an int that simultaneously carries provenance for
// both globals. Anything downstream must NOT be tagged with a single scope,
// otherwise we'd be telling LLVM "no alias to @B" about a load that may very
// well land in @B's region.
// -----------------------------------------------------------------------------

// CHECK-LABEL: gpu.module @ambiguous_add
gpu.module @ambiguous_add {
  llvm.mlir.global external @amb_a() {addr_space = 3 : i32, alignment = 1024 : i64, dso_local} : !llvm.array<0 x i8>
  llvm.mlir.global external @amb_b() {addr_space = 3 : i32, alignment = 1024 : i64, dso_local} : !llvm.array<0 x i8>

  // CHECK-LABEL: llvm.func @load_ambiguous
  llvm.func @load_ambiguous(%c: i32) -> vector<4xi32> {
    %a = llvm.mlir.addressof @amb_a : !llvm.ptr<3>
    %b = llvm.mlir.addressof @amb_b : !llvm.ptr<3>
    %ai = llvm.ptrtoint %a : !llvm.ptr<3> to i32
    %bi = llvm.ptrtoint %b : !llvm.ptr<3> to i32
    %amb = llvm.add %ai, %bi : i32
    %off = llvm.add %amb, %c : i32
    %p = llvm.inttoptr %off : i32 to !llvm.ptr<3>
    // CHECK: llvm.load
    // CHECK-NOT: alias_scopes
    %v = llvm.load %p : !llvm.ptr<3> -> vector<4xi32>
    llvm.return %v : vector<4xi32>
  }
}

// -----

// -----------------------------------------------------------------------------
// `or`/`sub`/`xor` are NOT canonical pointer arithmetic via int. Even when
// they happen to be equivalent to `add` they can break provenance, so the
// pass refuses to forward through them.
// -----------------------------------------------------------------------------

// CHECK-LABEL: gpu.module @nontracked_op
gpu.module @nontracked_op {
  llvm.mlir.global external @nt_a() {addr_space = 3 : i32, alignment = 1024 : i64, dso_local} : !llvm.array<0 x i8>
  llvm.mlir.global external @nt_b() {addr_space = 3 : i32, alignment = 1024 : i64, dso_local} : !llvm.array<0 x i8>

  // CHECK-LABEL: llvm.func @load_via_or
  llvm.func @load_via_or(%mask: i32) -> vector<4xi32> {
    %a = llvm.mlir.addressof @nt_a : !llvm.ptr<3>
    %ai = llvm.ptrtoint %a : !llvm.ptr<3> to i32
    %off = llvm.or %ai, %mask : i32
    %p = llvm.inttoptr %off : i32 to !llvm.ptr<3>
    // CHECK: llvm.load
    // CHECK-NOT: alias_scopes
    %v = llvm.load %p : !llvm.ptr<3> -> vector<4xi32>
    llvm.return %v : vector<4xi32>
  }
}
