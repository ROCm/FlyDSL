# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Per-vendor JIT runtime subdirectories, registered by cmake/backends/<name>.cmake.
get_property(_dirs GLOBAL PROPERTY FLYDSL_BACKEND_RUNTIME_SUBDIRS)
foreach(_dir ${_dirs})
  add_subdirectory(${_dir})
endforeach()
