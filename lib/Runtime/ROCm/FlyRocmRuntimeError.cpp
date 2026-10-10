//===- FlyRocmRuntimeError.cpp - Shared ROCm runtime errors ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FlyRocmRuntimeError.h"

// First runtime error observed on this thread since the last
// flydslRuntimeTakeError(); 0 when none. Positive values are hipError_t,
// negative values are FlyDSL AOT lifecycle errors.
thread_local static int32_t lastError = 0;

void flydslRecordRuntimeError(int32_t error) {
  if (error && !lastError)
    lastError = error;
}

extern "C" FLYDSL_RUNTIME_API int32_t flydslRuntimeTakeError() {
  int32_t error = lastError;
  lastError = 0;
  return error;
}
