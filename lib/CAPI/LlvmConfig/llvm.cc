// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
//
// LLVM cl::opt runtime control.
// Linked into libFlyPythonCAPI.so to share the cl::opt registry with MLIR.
// Uses addOccurrence() so getNumOccurrences()-guarded options work correctly.

#include "LlvmConfig/llvm.h"

#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Triple.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

using namespace llvm;

namespace {

/// Find a registered cl::Option by name.  Returns nullptr if not found.
cl::Option *findOption(const std::string &name) {
  auto options = cl::getRegisteredOptions();
  auto it = options.find(name);
  return it == options.end() ? nullptr : it->second;
}

/// Check whether \p opt is a bool-valued option (ValueOptional / ValueDisallowed)
/// vs. a value-required option (int / string / enum).
/// LLVM is built without RTTI, so dynamic_cast is unavailable; we use the
/// ValueExpected flag as a proxy for the underlying C++ type.
bool isBoolOption(cl::Option *opt) { return opt->getValueExpectedFlag() != cl::ValueRequired; }

} // namespace

// ---------------------------------------------------------------------------
// C API
// ---------------------------------------------------------------------------

extern "C" {

bool flydslGetRegisterClassLayout(const char *target, const char *name, unsigned *bits,
                                  unsigned *count) {
  static std::once_flag initialized;
  std::call_once(initialized, [] {
    InitializeAllTargetInfos();
    InitializeAllTargetMCs();
  });
  Triple triple(std::string(target) + "-unknown-unknown");
  std::string error;
  const Target *backend = TargetRegistry::lookupTarget(triple, error);
  if (!backend)
    return false;
  std::unique_ptr<MCRegisterInfo> registers(backend->createMCRegInfo(triple));
  if (!registers)
    return false;
  for (const MCRegisterClass &rc : registers->regclasses()) {
    if (StringRef(registers->getRegClassName(&rc)) == name) {
      *bits = rc.getSizeInBits();
      *count = rc.getNumRegs();
      return true;
    }
  }
  return false;
}

__attribute__((visibility("default"))) int flydslSetLLVMOptionBool(const char *name, bool value,
                                                                   bool *oldValue) {
  cl::Option *opt = findOption(name);
  if (!opt)
    return 1;
  if (!isBoolOption(opt))
    return 2; // type mismatch: option is not bool
  auto *typed = static_cast<cl::opt<bool> *>(opt);
  if (oldValue)
    *oldValue = typed->getValue();
  opt->addOccurrence(1, name, value ? "true" : "false");
  return 0;
}

__attribute__((visibility("default"))) int flydslSetLLVMOptionInt(const char *name, int value,
                                                                  int *oldValue) {
  cl::Option *opt = findOption(name);
  if (!opt)
    return 1;
  if (isBoolOption(opt))
    return 2; // type mismatch: option is bool, not int
  auto *typed = static_cast<cl::opt<int> *>(opt);
  if (oldValue)
    *oldValue = typed->getValue();
  opt->addOccurrence(1, name, std::to_string(value));
  return 0;
}

__attribute__((visibility("default"))) int
flydslSetLLVMOptionStr(const char *name, const char *value, char **oldValue) {
  cl::Option *opt = findOption(name);
  if (!opt)
    return 1;
  if (isBoolOption(opt))
    return 2; // type mismatch: option is bool, not string
  auto *typed = static_cast<cl::opt<std::string> *>(opt);
  if (oldValue) {
    std::string prev = typed->getValue();
    *oldValue = static_cast<char *>(std::malloc(prev.size() + 1));
    std::memcpy(*oldValue, prev.c_str(), prev.size() + 1);
  }
  opt->addOccurrence(1, name, value);
  return 0;
}

__attribute__((visibility("default"))) void flydslFreeLLVMOptionStr(char *str) { std::free(str); }

} // extern "C"
