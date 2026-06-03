# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Per-vendor JIT runtime subdirectories.
if("rocdl" IN_LIST FLYDSL_BACKENDS)
  add_subdirectory(ROCm)
endif()
