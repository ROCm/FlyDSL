// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
//
// LLVM cl::opt runtime control.
// Compiled into libFlyPythonCAPI.so so that getRegisteredOptions() accesses
// the same cl::opt registry used by the MLIR compilation pipeline.
//
// addOccurrence() is used instead of setValue() because some LLVM passes
// check getNumOccurrences() to distinguish "explicitly set" from "default".
// See llvm/lib/CodeGen/MachineScheduler.cpp — enableMachineSchedDefaultSched().

#include "llvm.h"

#include "llvm/Support/CommandLine.h"

#include <cstdlib>
#include <cstring>
#include <string>

using namespace llvm;

namespace {

template <typename T> T setOption(const std::string &name, T value);

template <> bool setOption<bool>(const std::string &name, bool value) {
  auto options = cl::getRegisteredOptions();
  auto it = options.find(name);
  if (it == options.end())
    return false;
  auto *opt = static_cast<cl::opt<bool> *>(it->second);
  bool original = opt->getValue();
  it->second->addOccurrence(1, name, value ? "true" : "false");
  return original;
}

template <> int setOption<int>(const std::string &name, int value) {
  auto options = cl::getRegisteredOptions();
  auto it = options.find(name);
  if (it == options.end())
    return 0;
  auto *opt = static_cast<cl::opt<int> *>(it->second);
  int original = opt->getValue();
  it->second->addOccurrence(1, name, std::to_string(value));
  return original;
}

template <>
std::string setOption<std::string>(const std::string &name, std::string value) {
  auto options = cl::getRegisteredOptions();
  auto it = options.find(name);
  if (it == options.end())
    return "";
  auto *opt = static_cast<cl::opt<std::string> *>(it->second);
  std::string original = opt->getValue();
  it->second->addOccurrence(1, name, value);
  return original;
}

bool hasOption(const std::string &name) {
  return cl::getRegisteredOptions().find(name) != cl::getRegisteredOptions().end();
}

} // namespace

// ---------------------------------------------------------------------------
// C API
// ---------------------------------------------------------------------------

extern "C" {

__attribute__((visibility("default")))
int flydslSetLLVMOptionBool(const char *name, bool value, bool *oldValue) {
  if (!hasOption(name))
    return 1;
  bool prev = setOption<bool>(name, value);
  if (oldValue)
    *oldValue = prev;
  return 0;
}

__attribute__((visibility("default")))
int flydslSetLLVMOptionInt(const char *name, int value, int *oldValue) {
  if (!hasOption(name))
    return 1;
  int prev = setOption<int>(name, value);
  if (oldValue)
    *oldValue = prev;
  return 0;
}

__attribute__((visibility("default")))
int flydslSetLLVMOptionStr(const char *name, const char *value,
                           char **oldValue) {
  if (!hasOption(name))
    return 1;
  std::string prev = setOption<std::string>(name, value);
  if (oldValue) {
    *oldValue = static_cast<char *>(std::malloc(prev.size() + 1));
    std::memcpy(*oldValue, prev.c_str(), prev.size() + 1);
  }
  return 0;
}

__attribute__((visibility("default")))
void flydslFreeLLVMOptionStr(char *str) { std::free(str); }

} // extern "C"
