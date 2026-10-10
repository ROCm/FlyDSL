// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --verify-diagnostics

func.func @zipped_divide_rejects_oversized_tile(
    %layout: !fly.layout<(64) : (1)>) {
  %tiler = fly.static : !fly.tile<[32|4]>
  // expected-error @+2 {{ZippedDivideOp: divisor tile rank exceeds layout rank}}
  // expected-error @+1 {{failed to infer returned types}}
  %result = fly.zipped_divide(%layout, %tiler)
      : (!fly.layout<(64) : (1)>, !fly.tile<[32|4]>)
      -> !fly.layout<((32, 4), (2)) : ((1, 0), (32))>
  return
}
