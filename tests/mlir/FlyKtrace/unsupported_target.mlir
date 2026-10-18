// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

// RUN: %fly-opt %s --convert-fly-ktrace-to-rocdl --verify-diagnostics

// Every ROCDL target must support in-kernel timestamps, not only the first.
module attributes {gpu.container_module} {
  // expected-error @below {{not supported on target arch 'gfx1100'}}
  gpu.module @kernels [#rocdl.target<chip = "gfx942">, #rocdl.target<chip = "gfx1100">] {
    gpu.func @mixed(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "m"
      gpu.return
    }
  }
}
