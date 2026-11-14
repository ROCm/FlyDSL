// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef CONVERSION_FLYTOROCDL_FLYTOROCDL_H
#define CONVERSION_FLYTOROCDL_FLYTOROCDL_H

#include "mlir/Pass/Pass.h"

#include <optional>
#include <string>

namespace mlir {
#define GEN_PASS_DECL_CONVERTROCDLFASTMATHOPSPASS
#define GEN_PASS_DECL_FLYTOROCDLCONVERSIONPASS
#define GEN_PASS_DECL_FLYEMITGPUBINARYPASS
#define GEN_PASS_DECL_FLYROCDLCLUSTERATTRPASS
#include "flydsl/Conversion/FlyToROCDL/Passes.h.inc"

#ifdef FLYDSL_HAS_LLD_LIBRARY
namespace fly {
/// Link a relocatable AMDGPU object into an HSA code object with the in-process
/// LLD that `fly-emit-gpu-binary` uses.  Returns the linker diagnostics on
/// failure, std::nullopt on success.
std::optional<std::string> linkHsacoInProcess(StringRef objectPath, StringRef hsacoPath);
} // namespace fly
#endif // FLYDSL_HAS_LLD_LIBRARY
} // namespace mlir

#endif // CONVERSION_FLYTOROCDL_FLYTOROCDL_H
