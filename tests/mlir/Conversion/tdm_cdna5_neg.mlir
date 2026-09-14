// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: { %fly-opt --split-input-file %s 2>&1 || true; } | FileCheck %s

// Iteration is paid for out of the descriptor's own GROUP2 slots: `tensor_dim2_stride`
// becomes `global_addr_increment`, `tensor_dim3` becomes `lds_addr_increment`, and
// `tile_dim3` becomes `iterate_count`. A dim whose stride the descriptor no longer holds
// is not a dim, so an iterating descriptor has two -- the third axis is the one the
// iteration itself walks. The builder never reaches this: `foldModes` drops iteration
// rather than refusing the atom. Hand-written IR does, and must be refused.

// -----

// CHECK: TDM descriptor iteration takes dim 2's stride for its own, so it needs a descriptor of at most 2 dims, got 3
func.func private @bad_cdna5_iterate_rank3(
    %a: !fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [8, 4, 64], elem = f16, tensor2tdm = (1E2,1E1,1E0), iterCount = 8>, 16>)

// -----

// CHECK: TDM descriptor iteration takes dim 2's stride for its own, so it needs a descriptor of at most 2 dims, got 3
func.func private @bad_cdna5_iterate_rank3_store(
    %a: !fly.copy_atom<!fly_rocdl.cdna5.tensor_store<shape = [8, 4, 64], elem = f16, tensor2tdm = (1E2,1E1,1E0), iterCount = 8>, 16>)
