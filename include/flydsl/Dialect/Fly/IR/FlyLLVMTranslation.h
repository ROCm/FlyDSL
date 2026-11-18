// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLY_IR_FLYLLVMTRANSLATION_H
#define FLYDSL_DIALECT_FLY_IR_FLYLLVMTRANSLATION_H

namespace mlir {
class DialectRegistry;

namespace fly {
/// Register the LLVM IR offloading translation of `#fly.explicit_module` and
/// `#fly.aot_module`, which the Fly dialect promises.
void registerOffloadingLLVMTranslationInterfaceExternalModels(DialectRegistry &registry);
} // namespace fly
} // namespace mlir

#endif // FLYDSL_DIALECT_FLY_IR_FLYLLVMTRANSLATION_H
