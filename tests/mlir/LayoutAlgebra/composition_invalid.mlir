// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --verify-diagnostics

func.func @composition_rejects_tiler_deeper_than_layout(
    %layout: !fly.layout<(12, 6) : (1, 16)>) {
  %tiler = fly.static : !fly.tile<[(4) : (1)|(3) : (1)|(5) : (1)]>
  // expected-error @+2 {{CompositionOp: tiler rank exceeds layout rank}}
  // expected-error @+1 {{failed to infer returned types}}
  %result = fly.composition(%layout, %tiler)
      : (!fly.layout<(12, 6) : (1, 16)>, !fly.tile<[(4) : (1)|(3) : (1)|(5) : (1)]>)
      -> !fly.layout<(4, 3) : (1, 16)>
  return
}
