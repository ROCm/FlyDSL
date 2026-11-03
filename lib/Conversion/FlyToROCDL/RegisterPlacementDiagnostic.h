// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_CONVERSION_FLYTOROCDL_REGISTERPLACEMENTDIAGNOSTIC_H
#define FLYDSL_CONVERSION_FLYTOROCDL_REGISTERPLACEMENTDIAGNOSTIC_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DiagnosticInfo.h"

#include <cstdint>
#include <optional>
#include <string>

namespace llvm {
class Function;
} // namespace llvm

namespace mlir::fly {

constexpr llvm::StringLiteral RegisterDiagnosticsFlag = "flydsl.register-diagnostics";

int getRegisterPlacementDiagnosticKind();

/// A rejected placement request. `Request` identifies the lowered
/// fly.register_value when known, so the error can point at the DSL source.
class RegisterPlacementDiagnostic : public llvm::DiagnosticInfo {
  const llvm::Function &Fn;
  std::optional<uint64_t> Request;
  std::string Message;

public:
  RegisterPlacementDiagnostic(const llvm::Function &Fn, std::optional<uint64_t> Request,
                              std::string Message)
      : DiagnosticInfo(getRegisterPlacementDiagnosticKind(), llvm::DS_Error), Fn(Fn),
        Request(Request), Message(std::move(Message)) {}

  std::optional<uint64_t> getRequest() const { return Request; }
  void print(llvm::DiagnosticPrinter &DP) const override;

  static bool classof(const llvm::DiagnosticInfo *DI) {
    return DI->getKind() == getRegisterPlacementDiagnosticKind();
  }
};

} // namespace mlir::fly

#endif // FLYDSL_CONVERSION_FLYTOROCDL_REGISTERPLACEMENTDIAGNOSTIC_H
