# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

# Per-vendor JIT runtime subdirectories; each defines a FlyJitRuntime target.
if(FLYDSL_BUILD_ROCDL_STACK)
  add_subdirectory(ROCm)
endif()
