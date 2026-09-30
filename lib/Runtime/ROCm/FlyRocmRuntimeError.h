//===- FlyRocmRuntimeError.h - Shared ROCm runtime errors -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FLYDSL_RUNTIME_ROCM_FLYROCMRUNTIMEERROR_H
#define FLYDSL_RUNTIME_ROCM_FLYROCMRUNTIMEERROR_H

#include <cstdint>
#include <cstdio>

#include "hip/hip_runtime.h"

#if defined(FLYDSL_AOT_RUNTIME_EMBEDDED)
#define FLYDSL_RUNTIME_API __attribute__((visibility("hidden")))
#else
#define FLYDSL_RUNTIME_API __attribute__((visibility("default")))
#endif

__attribute__((visibility("hidden"))) void flydslRecordRuntimeError(int32_t error);

extern "C" FLYDSL_RUNTIME_API int32_t flydslRuntimeTakeError();

#define HIP_REPORT_IF_ERROR(expr)                                                                  \
  [](hipError_t result) {                                                                          \
    if (!result)                                                                                   \
      return;                                                                                      \
    flydslRecordRuntimeError(static_cast<int32_t>(result));                                        \
    const char *name = hipGetErrorName(result);                                                    \
    if (!name)                                                                                     \
      name = "<unknown>";                                                                          \
    fprintf(stderr, "'%s' failed with '%s'\n", #expr, name);                                       \
  }(expr)

#endif // FLYDSL_RUNTIME_ROCM_FLYROCMRUNTIMEERROR_H
